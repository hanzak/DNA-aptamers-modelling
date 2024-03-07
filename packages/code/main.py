import transformer
import config
import dataset 
import pickle
import os
from BucketDataLoader import BucketDataLoader
import numpy as np
import json

config_ = config.get_config()

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
        train, valid, test = dataset.data_split(data_for_transformer)

        with open(f"data/splits/train_{data_name}_struct.pkl", 'wb') as f:
            pickle.dump(train, f)
        with open(f"data/splits/valid_{data_name}_struct.pkl", 'wb') as f:
            pickle.dump(valid, f)
        with open(f"data/splits/test_{data_name}_struct.pkl", 'wb') as f:
            pickle.dump(test, f) 
    return train, valid, test

from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize

np.int = int

def objective_function(hyperparameters):
    config_['learning_rate'] = hyperparameters[0]
    config_['batch_size'] = int(hyperparameters[1])
    config_['dropout'] = int(hyperparameters[2])
    
    valid_loss = transformer.train_model(config_, train_dataloader, valid_dataloader, test_dataloader)
    
    return valid_loss

def hyperparam_tune(objective_function):
    hyperparameter_space = [
        Real(1e-6, 1e-3, prior='log-uniform', name='learning_rate'),
        Categorical([128, 256, 512], name='batch_size'),
        Real(0.1, 0.7, name='dropout')
    ]

    results = gp_minimize(
        func=objective_function,
        dimensions=hyperparameter_space,
        n_calls=50,  
        random_state=0
    )

    results_dict = {
        'best_hyperparameters': results.x,
        'best_validation_loss': results.fun,
        'all_hyperparameters': results.x_iters,
        'all_validation_losses': results.func_vals
    }

    with open('optimization_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
        

data_name = "250k"


train, valid, test = get_data_structures(data_name)
#train_dataloader = BucketDataLoader(train, config_)
#valid_dataloader = BucketDataLoader(valid, config_)
#test_dataloader = BucketDataLoader(test, config_)
#transformer.train_model(config_, train_dataloader, valid_dataloader, test_dataloader)

train_dataloader = BucketDataLoader(train, config_)
valid_dataloader = BucketDataLoader(valid, config_)
test_dataloader = BucketDataLoader(test, config_)
hyperparam_tune(objective_function)




"""

import torch
import transformer

config_ = config.get_config()
device = config_['device']

train, valid, test = get_data_structures(data_name)
test_dataloader = BucketDataLoader(test, config_)

model_path = 'packages/model/model_checkpoint/06-03-2024_033200_2p5M_model_checkpoint.pth'
model = transformer.Transformer(config_)  
model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.eval()  


with torch.no_grad():
    i=0
    for batch in test_dataloader:
        sq, mfe, structure, mask = batch
        sq, structure, mask = sq.to(device), structure.to(device), mask.to(device)
            
        mfes, structures, structures_prob = model(sq, mask)
        predicted_indices = torch.argmax(structures_prob, dim=-1)  
        
        if i%100==0:
            print(f"MFE test 0: {mfes[0]}, MFE real 0: {mfe[0]}")
            print(f"structure 0: {predicted_indices[0]}, real 0: {structure[0]}")
            i=0
        i+=1
"""