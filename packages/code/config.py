def get_config():
    return {
        "batch_size": 64,
        "num_epochs": 500,
        "learning_rate": 5e-5,
        "d_model": 128,
        "N": 2,
        "heads": 4,
        "dropout": 0.10,
        "d_ff": 256,
        "vocab_size": 5,
        "sq_len": 150,
        "max_len":150,
        "model_folder": "weights",
        "model_filename": "transformer_model_",
        "preload": None,
        "exp_name": "packages/model/runs/tmodel"
    }