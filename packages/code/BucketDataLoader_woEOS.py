from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import config

import random

class BucketDataLoader_woEOS(DataLoader):
    def __init__(self, dataset, config, shuffle=False):
        super().__init__(dataset, batch_size=config['batch_size'], shuffle=shuffle, collate_fn=self.collate_fn)
        self.config=config

    def collate_fn(self, batch):
        #batch.sort(key=lambda x: len(x[0]), reverse=False)

        sequences, mfe, structures, num_hairpins = zip(*batch)
        max_len = 51
        
        padded_sequences = []
        for sequence in sequences:
            sequence = self.config['SOS'] + sequence
            sequence_tensor = torch.tensor([{'@': 1, 'A': 2, 'C': 3, 'G': 4, 'T': 5}[nuc] for nuc in sequence], dtype=torch.long)
            padded_sequence = F.pad(sequence_tensor, (0, max_len - len(sequence)), value=self.config['pad_value'])
            padded_sequences.append(padded_sequence)
                        
        padded_structures = []
        for structure in structures:
            structure = self.config['SOS'] + structure
            structure_tensor = torch.tensor([{'@': 1, '(': 2, ')': 3, '.': 4}[symbol] for symbol in structure], dtype=torch.long)
            padded_structure= F.pad(structure_tensor, (0, max_len - len(structure)), value=self.config['pad_value'])
            padded_structures.append(padded_structure)

        padded_sequences = torch.stack(padded_sequences)
        padded_structures = torch.stack(padded_structures)

        mfe_tensor = torch.tensor(mfe)
        mfe_tensor = mfe_tensor.unsqueeze(1)

        num_hairpins_tensor = torch.tensor(num_hairpins)
        num_hairpins_tensor = num_hairpins_tensor.unsqueeze(1)
                
        return padded_sequences, mfe_tensor, padded_structures, num_hairpins_tensor

