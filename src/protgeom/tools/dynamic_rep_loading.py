import pickle
import numpy as np
from os import walk
import gc
from pathlib import Path
current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent.parent.parent


def load_representations(model_name, layer):
    filepath = root_dir / "data" / "reps"
    files = next(walk(filepath))[2]
    model_files = [f for f in files if model_name in f]
    reps = []
    for file_ in model_files:
        print(file_)
        with open(filepath / file_, 'rb') as handle:
            model_reps = pickle.load(handle)
            reps.append(model_reps[layer])
        del model_reps
        gc.collect()
    flat_reps = [item for sublist in reps for item in sublist]
    return flat_reps

