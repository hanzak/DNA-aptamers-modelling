def get_config():
    return {
        "batch_size": 64,
        "num_epochs": 500,
        "learning_rate": 5e-5,
        "d_model": 512,
        "N": 3,
        "heads": 4,
        "dropout": 0.10,
        "d_ff": 1024,
        "vocab_size": 5,
        "sq_len": 200,
        "max_len":200,
        "preload": None,
        "exp_name": "packages/model/runs/tmodel",
        "prefix": "2p5M-"
    }