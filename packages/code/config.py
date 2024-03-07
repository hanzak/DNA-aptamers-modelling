import torch
def get_config():
    return {
        "batch_size": 256,
        "num_epochs": 80,
        "learning_rate": 1e-4,
        "d_model": 256,
        "layers_encoder": 2,
        "layers_decoder": 2,
        "heads": 4,
        "dropout": 0.2,
        "d_ff": 512,
        "src_vocab_size": 5,
        "target_vocab_size": 4,
        "max_len":200,
        "preload": None,
        "exp_name": "packages/model/runs/tmodel",
        "device": torch.device('cuda' if torch.cuda.is_available() else "cpu"),
        "pad_value": 0
    }