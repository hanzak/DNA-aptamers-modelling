from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F

class BucketDataLoader(DataLoader):
    """
    Class made to load data into tensors, ready to be used for training

    Args:
        DataLoader (DataLoader): A DataLoader
    """
    def __init__(self, dataset, config, shuffle=False):
        super().__init__(dataset, batch_size=config['batch_size'], shuffle=shuffle, collate_fn=self.collate_fn)
        self.config=config

    def collate_fn(self, batch):
        sequences, mfe, structures, num_hairpins = zip(*batch)
        max_len_seq = max(len(seq) for seq in sequences)

        #If we have a Transformer with decoder, max length is 2 more, since we count SOS and EOS tokens
        if self.config['type']=="decoder":
            max_len_struct = max(len(s) for s in structures) + 2
        else:
            max_len_struct = max(len(s) for s in structures)
        
        padded_sequences = []
        for sequence in sequences:
            sequence_tensor = torch.tensor([{'A': 1, 'C': 2, 'G': 3, 'T': 4}[nuc] for nuc in sequence], dtype=torch.long)
            padded_sequence = F.pad(sequence_tensor, (0, max_len_seq - len(sequence)), value=self.config['pad_value']) #Pad sequences to the maximum sequence length in batch
            padded_sequences.append(padded_sequence)
                        
        padded_structures = []
        for structure in structures:
            #If we have transformer with decoder, add EOS and SOS tokens
            if self.config['type']=="decoder":
                structure = self.config['SOS'] + structure + self.config['EOS'] 
                structure_tensor = torch.tensor([{'@': 1, '(': 2, ')': 3, '.': 4, '$': 5}[symbol] for symbol in structure], dtype=torch.long)
            else:
                structure_tensor = torch.tensor([{'(': 1, ')': 2, '.': 3}[symbol] for symbol in structure], dtype=torch.long)
            padded_structure= F.pad(structure_tensor, (0, max_len_struct - len(structure)), value=self.config['pad_value']) #Pad structures to the maximum structure length in batch
            padded_structures.append(padded_structure)

        padded_sequences = torch.stack(padded_sequences)
        padded_structures = torch.stack(padded_structures)
        mfe_tensor = torch.tensor(mfe)
        mfe_tensor = mfe_tensor.unsqueeze(1)
        num_hairpins_tensor = torch.tensor(num_hairpins)
        num_hairpins_tensor = num_hairpins_tensor.unsqueeze(1)
                
        return padded_sequences, mfe_tensor, padded_structures, num_hairpins_tensor

