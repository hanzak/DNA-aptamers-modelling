import torch
def get_config():
    return {
        "batch_size": 128,
        "num_epochs": 50,
        "learning_rate": 2e-4,
        "d_model": 256,
        "layers_encoder": 2,
        "layers_decoder": 2,
        "heads": 4,
        "dropout": 0.1,
        "d_ff": 512,
        "src_vocab_size": 5,
        "target_vocab_size": 4,
        "max_len":100,
        "preload": None,
        "exp_name": "packages/model/runs/",
        "device": torch.device('cuda' if torch.cuda.is_available() else "cpu"),
        "pad_value": 0,
        "data_size": "250k"
    }