from torch.utils.data import random_split
import torch
import numpy as np
import random
from torch.utils.data import Dataset, Subset, ConcatDataset

def data_split(data):
    dataset_size = len(data)

    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset
    
def count_hairpins(data):
    hairpins = []
    start_checking = False
    start_checking2 = False
    for _, _, struct in data:
        len_hairpin = 0
        hairpin = []
        for i in range(len(struct)):
            if start_checking2:
                if struct[i]=="(":
                    start_checking2=False
                    len_hairpin = 0
                elif struct[i]==")":
                    hairpin.append(len_hairpin)
                    len_hairpin = 0
                    start_checking2=False
                    start_checking=False
            if start_checking:
                if struct[i]==".":
                    len_hairpin += 1
                    start_checking2=True
            if struct[i]=="(":
                start_checking=True


        hairpins.append(hairpin)
    
    num_hairpins = []
    for s in hairpins:
        num_hairpins.append(len(s))
            
    new_data = [(sq, mfe, struct, num_hairpins[i]) for i, (sq, mfe, struct) in enumerate(data)]
    
    return new_data


        

