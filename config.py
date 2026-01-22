from pathlib import Path

def get_config():
    return {
        "batch_size": 4,
        "num_epochs": 30,
        "lr": 10**-6,
        "seq_len": 256,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "ru",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "19",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "use_autocast": False,
        "num_workers": 2,
        "log_every": 50,
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)