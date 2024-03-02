import torch
def get_config():
    return {
        "batch_size": 256,
        "num_epochs": 500,
        "learning_rate": 1e-3,
        "d_model": 128,
        "N": 2,
        "heads": 4,
        "dropout": 0.11,
        "d_ff": 256,
        "vocab_size": 5,
        "max_len":200,
        "preload": None,
        "exp_name": "packages/model/runs/tmodel",
        "prefix": "2p5M-",
        "device": torch.device('cuda' if torch.cuda.is_available() else "cpu"),
        "pad_value": 0
    }