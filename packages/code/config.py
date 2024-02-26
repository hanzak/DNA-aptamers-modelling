def get_config():
    return {
        "batch_size": 128,
        "num_epochs": 300,
        "learning_rate": 1e-4,
        "d_model": 256,
        "N": 2,
        "heads": 4,
        "dropout": 0.1,
        "d_ff": 1024,
        "vocab_size": 5,
        "sq_len": 50,
        "max_len":50,
        "model_folder": "weights",
        "model_filename": "transformer_model_",
        "preload": None,
        "exp_name": "../model/runs/tmodel"
    }