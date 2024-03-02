from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F

class BucketDataLoader(DataLoader):
    def __init__(self, dataset, config, shuffle=False):
        super().__init__(dataset, batch_size=config['batch_size'], shuffle=shuffle, collate_fn=self.collate_fn)
        self.config=config

    def collate_fn(self, batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)

        sequences, mfe = zip(*batch)
        max_len = len(sequences[0])

        padded_sequences = []
        for sequence in sequences:
            sequence_tensor = torch.tensor([{'A': 1, 'C': 2, 'G': 3, 'T': 4}[nuc] for nuc in sequence], dtype=torch.long)
            padded_sequence = F.pad(sequence_tensor, (0, max_len - len(sequence_tensor)), value=self.config['pad_value'])
            padded_sequences.append(padded_sequence)
            
        padded_sequences = torch.stack(padded_sequences)
        mfe_tensor = torch.tensor(mfe)
        mfe_tensor = mfe_tensor.unsqueeze(1)
        mask = (padded_sequences == self.config['pad_value'])

        return padded_sequences, mfe_tensor, mask

