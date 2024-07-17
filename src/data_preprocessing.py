import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import glob

def download_and_extract_maestro(data_dir='data/maestro-v3.0.0'):
    data_dir = pathlib.Path(data_dir)
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'maestro-v3.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip',
            extract=True,
            cache_dir='.', 
            cache_subdir='data',
        )
        extracted_dir = pathlib.Path('./data/maestro-v3.0.0')
        extracted_dir.rename(data_dir)
    print(f"Dataset is ready in {data_dir}")

def count_midi_files(data_dir='data/maestro-v3.0.0'):
    data_dir = pathlib.Path(data_dir)
    files = glob.glob(str(data_dir/'**/*.mid*'))
    print('Number of MIDI files:', len(files))
    return len(files)

if __name__ == "__main__":
    download_and_extract_maestro()
    count_midi_files()