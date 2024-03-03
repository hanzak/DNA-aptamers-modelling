import transformer
import config
import dataset 
import pickle
import os
from BucketDataLoader import BucketDataLoader
import numpy as np

train_path = 'packages/data/train_2p5M.pkl'
valid_path = 'packages/data/valid_2p5M.pkl'
test_path = 'packages/data/test_2p5M.pkl'

def check_file(file_path):
    return os.path.isfile(file_path) and os.stat(file_path).st_size > 0

if check_file(train_path) and check_file(valid_path) and check_file(test_path):
    with open(train_path, 'rb') as f:
        train = pickle.load(f)
    with open(valid_path, 'rb') as f:
        valid = pickle.load(f)
    with open(test_path, 'rb') as f:
        test = pickle.load(f)
else:
    if check_file("data_2p5M.pkl") == False:
        raise FileNotFoundError("data_2p5M.pkl doesn't exist or is empty")
    with open('data_2p5M.pkl', 'rb') as file:
        data_for_transformer = pickle.load(file)
    train, valid, test = dataset.data_split(data_for_transformer)
    
    with open(train_path, 'wb') as f:
        pickle.dump(train, f)
    with open(valid_path, 'wb') as f:
        pickle.dump(valid, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test, f) 

from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize

np.int = int

def objective_function(hyperparameters):
    config_ = config.get_config()
    print(type(hyperparameters[5]))
    config_['layers_encoder'] = hyperparameters[0]
    config_['heads'] = hyperparameters[1]
    config_['d_model'] = hyperparameters[2]
    config_['d_ff'] = hyperparameters[3]
    config_['learning_rate'] = hyperparameters[4]
    config_['batch_size'] = int(hyperparameters[5])
    
    train_dataloader = BucketDataLoader(train, config_)
    valid_dataloader = BucketDataLoader(valid, config_)
    test_dataloader = BucketDataLoader(test, config_)
    
    valid_loss = transformer.train_model(config_, train_dataloader, valid_dataloader, test_dataloader)
    
    return valid_loss

def hyperparam_tune(objective_function):
    hyperparameter_space = [
        Integer(1, 3, name='layers_encoder'),  
        Categorical([4, 8], name='heads'),  
        Categorical([128,256,512], name='d_model'),  
        Categorical([256,512,1024], name='d_ff'),  
        Real(1e-5, 1e-2, "log-uniform", name='learning_rate'),  
        Categorical([64, 128, 256, 512], name='batch_size') 
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


