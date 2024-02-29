def get_config():
    return {
        "batch_size": 256,
        "num_epochs": 25,
        "learning_rate": 5e-5,
        "d_model": 128,
        "N": 4,
        "heads": 8,
        "dropout": 0.1,
        "d_ff": 512,
        "vocab_size": 5,
        "sq_len": 200,
        "max_len":200,
        "model_folder": "weights",
        "model_filename": "transformer_model_",
        "preload": None,
        "exp_name": "packages/model/runs/tmodel"
    }