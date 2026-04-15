import os
from src.helper import load_audio, save_audio, add_white_noise
import matplotlib.pyplot as plt
import pywt
import numpy as np

def main():
    input_file = "data/raw/clean_sample.wav"
    noisy_out = "data/raw/noisy_test.wav"
    processed_out = "data/processed/denoised_test.wav"
    if not(os.path.exists(input_file)):
        print(f"Wav file is missing at {input_file}")
        return

    print(f"loading {input_file}")
    fs, clean_data = load_audio(input_file)
    noisy_data = add_white_noise(clean_data, noise_level = .02)
    save_audio(noisy_out, fs, noisy_data)
    print(f"Noisy test file is made at {noisy_out}")

    # Decompose the signal
    wavelet_type = 'db4'
    decomposed_coeffs = pywt.wavedec(noisy_data, wavelet_type, level=4)
    
    # Denoise signal
    for i in range(2, len(decomposed_coeffs)):
        decomposed_coeffs[i] = pywt.threshold(decomposed_coeffs[i], 0.1, 'soft')
    
    # Reconstruct signal
    denoised_data = pywt.waverec(decomposed_coeffs, wavelet_type)
    
    save_audio(processed_out, fs, denoised_data)
    print(f"Saved cleaned audio to {processed_out}")

if __name__ == "__main__":
    main()