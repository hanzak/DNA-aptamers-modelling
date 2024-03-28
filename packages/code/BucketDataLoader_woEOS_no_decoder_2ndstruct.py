from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F

class BucketDataLoader_woEOS_no_decoder_2ndstruct(DataLoader):
    def __init__(self, dataset, config, shuffle=False):
        super().__init__(dataset, batch_size=config['batch_size'], shuffle=shuffle, collate_fn=self.collate_fn)
        self.config=config

    def collate_fn(self, batch):

        sequences, mfe, struct, num_hairpins = zip(*batch)
        max_len = max(len(s) for s in struct)

        padded_structures = []
        for structure in struct:
            structure_tensor = torch.tensor([{'(': 1, ')': 2, '.': 3}[nuc] for nuc in structure], dtype=torch.long)
            padded_structure = F.pad(structure_tensor, (0, max_len - len(structure_tensor)), value=self.config['pad_value'])
            padded_structures.append(padded_structure)
            
        padded_structures = torch.stack(padded_structures)
        mfe_tensor = torch.tensor(mfe)
        mfe_tensor = mfe_tensor.unsqueeze(1)
        num_hairpins_tensor = torch.tensor(num_hairpins)
        num_hairpins_tensor = num_hairpins_tensor.unsqueeze(1)        

        return padded_structures, mfe_tensor, num_hairpins_tensor

