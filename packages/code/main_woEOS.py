import transformer_woEOS
import config
import dataset 
import pickle
import os
import torch
from BucketDataLoader_woEOS import BucketDataLoader_woEOS
import numpy as np
import json
import random
from NpEncoder import *
from torch.utils.data import Dataset, ConcatDataset

config_ = config.get_config()

best_validation_loss = float(1e9)
best_model_path = None

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def check_file(file_path):
    return os.path.isfile(file_path) and os.stat(file_path).st_size > 0


def get_data_mfes():
    if check_file(train_path_mfe) and check_file(valid_path_mfe) and check_file(test_path_mfe):
        with open(train_path_mfe, 'rb') as f:
            train = pickle.load(f)
        with open(valid_path_mfe, 'rb') as f:
            valid = pickle.load(f)
        with open(test_path_mfe, 'rb') as f:
            test = pickle.load(f)
    else:
        if check_file("data/data_250k.pkl") == False:
            raise FileNotFoundError("data_5M.pkl doesn't exist or is empty")
        with open('data/data_250k.pkl', 'rb') as file:
            data_for_transformer = pickle.load(file)
        train, valid, test = dataset.data_split(data_for_transformer)

        with open(train_path_mfe, 'wb') as f:
            pickle.dump(train, f)
        with open(valid_path_mfe, 'wb') as f:
            pickle.dump(valid, f)
        with open(test_path_mfe, 'wb') as f:
            pickle.dump(test, f) 
            
    return train, valid, test


def get_data_structures(data_name):
    if check_file(f"data/splits/train_{data_name}_struct.pkl") and check_file(f"data/splits/valid_{data_name}_struct.pkl") and check_file(f"data/splits/test_{data_name}_struct.pkl"):
        with open(f"data/splits/train_{data_name}_struct.pkl", 'rb') as f:
            train = pickle.load(f)
        with open(f"data/splits/valid_{data_name}_struct.pkl", 'rb') as f:
            valid = pickle.load(f)
        with open(f"data/splits/test_{data_name}_struct.pkl", 'rb') as f:
            test = pickle.load(f)
    else:
        if check_file(f"data/data_{data_name}_struct.pkl") == False:
            raise FileNotFoundError(f"data_{data_name}_struct.pkl doesn't exist or is empty")
        with open(f'data/data_{data_name}_struct.pkl', 'rb') as file:
            data_for_transformer = pickle.load(file)
                    
        data_for_transformer = dataset.count_hairpins(data_for_transformer)
        
        train, valid, test = dataset.data_split(data_for_transformer)

        with open(f"data/splits/train_{data_name}_struct.pkl", 'wb') as f:
            pickle.dump(train, f)
        with open(f"data/splits/valid_{data_name}_struct.pkl", 'wb') as f:
            pickle.dump(valid, f)
        with open(f"data/splits/test_{data_name}_struct.pkl", 'wb') as f:
            pickle.dump(test, f) 
    return train, valid, test

"""
def get_data_with_priming(data_name):
    if check_file(f"data/data_{data_name}_struct.pkl") == False:
        raise FileNotFoundError(f"data_{data_name}_struct.pkl doesn't exist or is empty")
    with open(f'data/data_{data_name}_struct.pkl', 'rb') as file:
        data_for_transformer = pickle.load(file)
            
    if check_file(f"data/priming/data_priming_2p5M.pkl") == False:
        raise FileNotFoundError(f"data_priming_{data_name}.pkl doesn't exist or is empty")
    with open(f'data/priming/data_priming_2p5M.pkl', 'rb') as pfile:
        priming = pickle.load(pfile)
        
    priming = dataset.count_hairpins(priming)
    data_for_transformer = dataset.count_hairpins(data_for_transformer)
        
    train, valid, test = dataset.data_split(data_for_transformer)
    
    priming_dataset = MyDataset(priming)
    
    updated_train = ConcatDataset([train.dataset, priming_dataset])
    
    return updated_train, valid, test
"""

from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize

np.int = int

def objective_function(hyperparameters):
    global best_validation_loss
    global best_model_path
    
    config_['learning_rate'] = hyperparameters[0]
    config_['batch_size'] = int(hyperparameters[1])
    config_['dropout'] = hyperparameters[2]
    
    train_dataloader = BucketDataLoader_woEOS(train, config_)
    valid_dataloader = BucketDataLoader_woEOS(valid, config_)
    test_dataloader = BucketDataLoader_woEOS(test, config_)
    
    valid_loss, model_path = transformer_woEOS.train_model(config_, train_dataloader, valid_dataloader)
    
    if valid_loss < best_validation_loss:
        best_validation_loss = valid_loss
        best_model_path = model_path
    
    return valid_loss

def hyperparam_tune(objective_function):
    hyperparameter_space = [
        Real(1e-5, 1e-3, prior='log-uniform', name='learning_rate'),
        Categorical([128, 256, 512], name='batch_size'),
        Real(0.1, 0.4, name='dropout')
    ]

    results = gp_minimize(
        func=objective_function,
        dimensions=hyperparameter_space,
        n_calls=25,  
        random_state=42
    )

    results_dict = {
        'best_hyperparameters': results.x,
        'best_validation_loss': results.fun,
        'best_model_path': best_model_path,
        'all_hyperparameters': results.x_iters,
        'all_validation_losses': results.func_vals
    }

    with open('packages/model/runs/tmodel/tuning/optimization_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4, cls=NpEncoder)

        
"""
def mock_train_model(config, train_dataloader, valid_dataloader):
    valid_loss = random.uniform(0, 1)
    model_path = f"dummy_model_{valid_loss}.pth"
    return valid_loss, model_path
"""
        

data_name = "7p5M"

config_['data_size'] = data_name

train, valid, test = get_data_structures(data_name)

train_dataloader = BucketDataLoader_woEOS(train, config_)
valid_dataloader = BucketDataLoader_woEOS(valid, config_)

transformer_woEOS.train_model(config_, train_dataloader, valid_dataloader)


#hyperparam_tune(objective_function)

"""
with open(f'data/data_generalization.pkl', 'rb') as file:
    data_gen = pickle.load(file)
    
new_data = []
testing=True
l=50
r=61
while testing:
    if r == 101:
        testing=False
    for d in data_gen:
        sq,_,_ = d
        if len(sq)>l  and len(sq)<r:
            new_data.append(d)
    new_data = dataset.count_hairpins(new_data)
        
    test_dataloader = BucketDataLoader_woEOS(new_data, config_)
    pred, act = transformer_woEOS.evaluate_model(config_, test_dataloader, "packages/model/model_checkpoint/2p5M/2p5M_woEOS_model_checkpoint.pth")
    
    new_data=[]
    
    l+=10
    r+=10
"""

#for i in range (len(pred)):
   # print(f"pred: {pred[i]}, act: {act[i]}")