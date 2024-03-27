import transformer_woEOS_no_decoder
import config_no_decoder
import dataset 
import pickle
import os
import torch
from BucketDataLoader_woEOS_no_decoder import BucketDataLoader_woEOS_no_decoder
import numpy as np
import json
import random
from NpEncoder import *
from torch.utils.data import Dataset, ConcatDataset

config_ = config_no_decoder.get_config()

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

data_name = "7p5M"

config_['data_size'] = data_name

train, valid, test = get_data_structures(data_name)

train_dataloader = BucketDataLoader_woEOS_no_decoder(train, config_)
valid_dataloader = BucketDataLoader_woEOS_no_decoder(valid, config_)
test_dataloader = BucketDataLoader_woEOS_no_decoder(test, config_)


transformer_woEOS_no_decoder.train_model(config_, train_dataloader, valid_dataloader)

"""
with open(f'data/data_generalization.pkl', 'rb') as file:
    data_gen = pickle.load(file)

new_data = []

for d in data_gen:
    sq,_,_ = d
    if len(sq)>50  and len(sq)<61:
        new_data.append(d)
new_data = dataset.count_hairpins(new_data)
        
test_dataloader = BucketDataLoader_woEOS_no_decoder(new_data, config_)
"""
"""
with open(f'data/data_generalization.pkl', 'rb') as file:
    data_gen = pickle.load(file)
new_data = []
for d in data_gen:
    sq,_,_ = d
    if len(sq)==100:
        new_data.append(d)
    if len(new_data)==100:
        break
new_data = dataset.count_hairpins(new_data)
test_dataloader = BucketDataLoader_woEOS_no_decoder(new_data, config_)
"""
#pred, act = transformer_woEOS_no_decoder.evaluate_model(config_, test_dataloader, "packages/model/model_checkpoint/2p5M/67829-2p5M_weEOS_no-decoder_complex_500len_model_checkpoint.pth")
#pred, act = transformer_woEOS_no_decoder.evaluate_model(config_, test_dataloader, "packages/model/model_checkpoint/5M/72865-5M_weEOS_no-decoder_complex_model_checkpoint.pth")
#pred, act = transformer_woEOS_no_decoder.evaluate_model(config_, test_dataloader, "packages/model/model_checkpoint/2p5M/75131-2p5M_weEOS_no-decoder_complex_model_checkpoint.pth")


    
        



