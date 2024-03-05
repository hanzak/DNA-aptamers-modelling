import transformer
import config
import dataset 
import pickle
import os
from BucketDataLoader import BucketDataLoader
import numpy as np

train_path_mfe = 'data/splits/train_2p5M.pkl'
valid_path_mfe = 'data/splits/valid_2p5M.pkl'
test_path_mfe = 'data/splits/test_2p5M.pkl'

train_path_struct = 'data/splits/train_2p5M_struct.pkl'
valid_path_struct = 'data/splits/valid_2p5M_struct.pkl'
test_path_struct = 'data/splits/test_2p5M_struct.pkl'

config_ = config.get_config()

def check_file(file_path):
    return os.path.isfile(file_path) and os.stat(file_path).st_size > 0

"""
def get_data_mfe():
    if check_file(train_path_mfe) and check_file(valid_path_mfe) and check_file(test_path_mfe):
        with open(train_path_mfe, 'rb') as f:
            train = pickle.load(f)
        with open(valid_path_mfe, 'rb') as f:
            valid = pickle.load(f)
        with open(test_path_mfe, 'rb') as f:
            test = pickle.load(f)
    else:
        if check_file("data_2p5M.pkl") == False:
            raise FileNotFoundError("data_2p5M.pkl doesn't exist or is empty")
        with open('data_2p5M.pkl', 'rb') as file:
            data_for_transformer = pickle.load(file)
        train, valid, test = dataset.data_split(data_for_transformer)

        with open(train_path_mfe, 'wb') as f:
            pickle.dump(train, f)
        with open(valid_path_mfe, 'wb') as f:
            pickle.dump(valid, f)
        with open(test_path_mfe, 'wb') as f:
            pickle.dump(test, f) 
            
    return train, valid, test
"""

def get_data_structures():
    if check_file(train_path_struct) and check_file(valid_path_struct) and check_file(test_path_struct):
        with open(train_path_struct, 'rb') as f:
            train = pickle.load(f)
        with open(valid_path_struct, 'rb') as f:
            valid = pickle.load(f)
        with open(test_path_struct, 'rb') as f:
            test = pickle.load(f)
    else:
        if check_file("data/data_2p5M_struct.pkl") == False:
            raise FileNotFoundError("data_2p5M_struct.pkl doesn't exist or is empty")
        with open('data/data_2p5M_struct.pkl', 'rb') as file:
            data_for_transformer = pickle.load(file)
        #print(len(data_for_transformer))
        #mfes = [mfe for _, mfe, _ in data_for_transformer]
        #standardized_mfes = dataset.standardize_mfe(mfes)
        #standardized_data = [(sequence, standardized_mfe, structure) for (sequence, _, structure), standardized_mfe in zip(data_for_transformer, standardized_mfes)]
        train, valid, test = dataset.data_split(data_for_transformer)

        with open(train_path_struct, 'wb') as f:
            pickle.dump(train, f)
        with open(valid_path_struct, 'wb') as f:
            pickle.dump(valid, f)
        with open(test_path_struct, 'wb') as f:
            pickle.dump(test, f) 
    return train, valid, test

from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize

np.int = int

def objective_function(hyperparameters):
    config_ = config.get_config()
    config_['layers_encoder'] = hyperparameters[0]
    config_['layers_decoder'] = hyperparameters[1]
    config_['heads'] = hyperparameters[2]
    config_['d_model'] = hyperparameters[3]
    config_['d_ff'] = hyperparameters[4]
    config_['learning_rate'] = hyperparameters[5]
    config_['batch_size'] = int(hyperparameters[6])
    
    train_dataloader = BucketDataLoader(train, config_)
    valid_dataloader = BucketDataLoader(valid, config_)
    test_dataloader = BucketDataLoader(test, config_)
    
    valid_loss = transformer.train_model(config_, train_dataloader, valid_dataloader, test_dataloader)
    
    return valid_loss

def hyperparam_tune(objective_function):
    hyperparameter_space = [
        Integer(1, 2, name='layers_encoder'),  
        Integer(1, 2, name='layers_decoder'),
        Categorical([2, 4], name='heads'),  
        Categorical([128, 256], name='d_model'),  
        Categorical([256, 512], name='d_ff'),  
        Categorical([1e-6, 1e-5, 1e-4, 1e-3], name='learning_rate'),  
        Categorical([256, 512], name='batch_size'),
        Categorical([0.1, 0.15, 0.2], name='dropout')
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
        

train, valid, test = get_data_structures()
train_dataloader = BucketDataLoader(train, config_)
valid_dataloader = BucketDataLoader(valid, config_)
test_dataloader = BucketDataLoader(test, config_)
transformer.train_model(config_, train_dataloader, valid_dataloader, test_dataloader)

#hyperparam_tune(objective_function)

"""
import torch
import transformer

config_ = config.get_config()
device = config_['device']

train, valid, test = get_data_structures()
test_dataloader = BucketDataLoader(test, config_)

model_path = 'packages/model/runs/tmodel/tuning/transformer_04-03-2024_161833.pth'
model = transformer.Transformer(config_)  
model.load_state_dict(torch.load(model_path))
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