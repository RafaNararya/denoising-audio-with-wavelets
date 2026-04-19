#Helper file for input/output stuff
import numpy as np
from scipy.io import wavfile 
import os
import cv2 

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

def add_image_noise(image, sigma=0.15, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def load_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0

def save_image(file_path, data):
    out_data = (np.clip(data, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(file_path, out_data)