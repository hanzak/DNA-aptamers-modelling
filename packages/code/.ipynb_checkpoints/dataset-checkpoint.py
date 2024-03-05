from torch.utils.data import random_split
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

def data_split(data):
    dataset_size = len(data)

    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset


def standardize_mfe(mfes):
    mfes_ = [[mfe] for mfe in mfes]
    
    scaler = StandardScaler()
    standardized_mfes = scaler.fit_transform(mfes_)
    
    standardized_mfes = [mfe[0] for mfe in standardized_mfes]
    
    return standardized_mfes
        


