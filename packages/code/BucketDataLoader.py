from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import config

import random

class BucketDataLoader(DataLoader):
    def __init__(self, dataset, config, shuffle=False):
        super().__init__(dataset, batch_size=config['batch_size'], shuffle=shuffle, collate_fn=self.collate_fn)
        self.config=config

    def collate_fn(self, batch):
        #batch.sort(key=lambda x: len(x[0]), reverse=True)
        sequences, mfe, structures, num_hairpins = zip(*batch)
        
        max_len = self.config['max_len']
        sq_max_len = 50
        
        random_augmentation_batch = random.randint(sq_max_len, max_len)
        
        padded_sequences = []
        padded_structures = []
        for i, sequence in enumerate(sequences):
            augment_chance = random.random()
            
            sq_len = len(sequence)
            
            if augment_chance > 0.8 and random_augmentation_batch!=sq_max_len:
                shift_value = random.randint(1, random_augmentation_batch-sq_len)
            else:
                shift_value = 0
                
            pre_pad = shift_value
            post_pad = random_augmentation_batch-pre_pad-sq_len
            
            sequence_tensor = torch.tensor([{'A': 1, 'C': 2, 'G': 3, 'T': 4}[nuc] for nuc in sequence], dtype=torch.long)
            structure_tensor = torch.tensor([{'(': 1, ')': 2, '.': 3}[symbol] for symbol in structures[i]], dtype=torch.long)
            
            padded_sequence = F.pad(sequence_tensor, (pre_pad, post_pad), value=self.config['pad_value'])
            padded_structure = F.pad(structure_tensor, (pre_pad, post_pad), value=self.config['pad_value'])
                        
            padded_sequences.append(padded_sequence)
            padded_structures.append(padded_structure)      
            
        padded_sequences = torch.stack(padded_sequences)
        padded_structures = torch.stack(padded_structures)
        
        mfe_tensor = torch.tensor(mfe)
        mfe_tensor = mfe_tensor.unsqueeze(1)
        
        num_hairpins_tensor = torch.tensor(num_hairpins)
        num_hairpins_tensor = num_hairpins_tensor.unsqueeze(1)
                
        mask = (padded_sequences == self.config['pad_value']).bool()
                
        return padded_sequences, mfe_tensor, padded_structures, num_hairpins_tensor, mask

