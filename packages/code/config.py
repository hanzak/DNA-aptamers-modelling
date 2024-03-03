import torch
def get_config():
    return {
        "batch_size": 256,
        "num_epochs": 200,
        "learning_rate": 1e-3,
        "d_model": 256,
        "layers_encoder": 3,
        "layers_decoder": 3,
        "heads": 4,
        "dropout": 0.10,
        "d_ff": 512,
        "src_vocab_size": 5,
        "target_vocab_size": 3,
        "max_len":200,
        "preload": None,
        "exp_name": "packages/model/runs/tmodel/tuning",
        "prefix": "2p5M-",
        "device": torch.device('cuda' if torch.cuda.is_available() else "cpu"),
        "pad_value": 0
    }