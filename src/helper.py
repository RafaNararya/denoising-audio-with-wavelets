#Helper file for input/output stuff
import numpy as np
from scipy.io import wavfile 
import os

def load_audio(file_path):
    fs, data = wavfile.read(file_path)
    if (data.dtype == np.int16):
        data = data.astype(np.float32) / 32768.0
    if len(data.shape) > 1:
        data = np.mean(data, axis = 1)

    return fs, data

def save_audio(file_path, fs, data):
    out_data = (data * 32767).astype(np.int16)
    wavfile.write(file_path, fs, out_data)

def add_white_noise(data, noise_level = .05, seed=None):
    np.random.seed(seed)
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise