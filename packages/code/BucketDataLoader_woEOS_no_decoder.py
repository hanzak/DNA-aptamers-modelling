from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F

class BucketDataLoader_woEOS_no_decoder(DataLoader):
    def __init__(self, dataset, config, shuffle=False):
        super().__init__(dataset, batch_size=config['batch_size'], shuffle=shuffle, collate_fn=self.collate_fn)
        self.config=config

    def collate_fn(self, batch):

        sequences, mfe, struct, num_hairpins = zip(*batch)
        max_len = max(len(seq) for seq in sequences) + 1

        padded_sequences = []
        for sequence in sequences:
            sequence_tensor = torch.tensor([{'A': 1, 'C': 2, 'G': 3, 'T': 4}[nuc] for nuc in sequence], dtype=torch.long)
            padded_sequence = F.pad(sequence_tensor, (0, max_len - len(sequence_tensor)), value=self.config['pad_value'])
            padded_sequences.append(padded_sequence)
            
        padded_sequences = torch.stack(padded_sequences)
        mfe_tensor = torch.tensor(mfe)
        mfe_tensor = mfe_tensor.unsqueeze(1)
        num_hairpins_tensor = torch.tensor(num_hairpins)
        num_hairpins_tensor = num_hairpins_tensor.unsqueeze(1)        

        return padded_sequences, mfe_tensor, num_hairpins_tensor

