import torch
def get_config():
    return {
        "batch_size": 256,
        "num_epochs": 200,
        "learning_rate": 1e-4,
        "d_model": 128,
        "layers_encoder": 2,
        "layers_decoder": 2,
        "heads": 4,
        "dropout": 0.11,
        "d_ff": 256,
        "src_vocab_size": 5,
        "target_vocab_size": 4,
        "max_len":200,
        "preload": None,
        "exp_name": "packages/model/runs/tmodel/tuning",
        "prefix": "2p5M-",
        "device": torch.device('cuda' if torch.cuda.is_available() else "cpu"),
        "pad_value": 0
    }