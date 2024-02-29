from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np
import config
import torch.nn.functional as F

class DNASequenceDataset(Dataset):
    def __init__(self, sequences, mfe_values, max_len, pad_value=0):
        self.sequences = sequences
        self.mfe_values = mfe_values
        self.pad_value = pad_value
        
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        mfe = self.mfe_values[idx]
        
        sequence_tensor = torch.tensor([{'A': 1, 'C': 2, 'G': 3, 'T': 4}[nuc] for nuc in sequence], dtype=torch.long)
        mfe_tensor = torch.tensor([mfe], dtype=torch.float)
        
        padded_sequence = F.pad(sequence_tensor, (0, self.max_len - len(sequence_tensor)), value=self.pad_value)
        mask = padded_sequence != self.pad_value
        
        return padded_sequence, mfe_tensor, mask

def data_split(data, config):
    filtered_dataset = [(seq, mfe) for seq, mfe in data if mfe != 0]

    dataset_size = len(filtered_dataset)

    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(filtered_dataset, [train_size, val_size, test_size])
    
    dna_sequences, mfe_values = zip(*train_dataset)
    #mfe_values_list = list(map(float, mfe_values))
    
    # Convert normalized values back to a tensor
    #mfe_norm_tensor = torch.tensor(mfe_norm)

    dataset = DNASequenceDataset(dna_sequences, mfe_values, config['max_len'])
    train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    dna_sequences, mfe_values = zip(*val_dataset)
    #mfe_values_list = list(map(float, mfe_values))

    # Convert normalized values back to a tensor
    #mfe_norm_tensor = torch.tensor(mfe_norm)

    dataset = DNASequenceDataset(dna_sequences, mfe_values, config['max_len'])
    val_dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    dna_sequences, mfe_values = zip(*test_dataset)
    #mfe_values_list = list(map(float, mfe_values))

    # Convert normalized values back to a tensor
    #mfe_norm_tensor = torch.tensor(mfe_norm)    

    dataset = DNASequenceDataset(dna_sequences, mfe_values, config['max_len'])
    test_dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader

def createDataLoader(data, batch_size, shuffle=False):
    return DataLoader(data, batch_size, shuffle)

