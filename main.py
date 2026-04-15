import os
from src.helper import load_audio, save_audio, add_white_noise
from src.thresholding import threshold
import matplotlib.pyplot as plt
import pywt
import numpy as np

def snr(clean, noisy):
    noise = noisy - clean
    p_signal = np.sum(clean**2)
    p_noise = np.sum(noise**2)
    return 10*np.log10(p_signal/p_noise)

def main():
    input_file = "data/raw/clean_sample.wav"
    noisy_out = "data/raw/noisy_test.wav"
    processed_out = "data/processed/denoised_test.wav"
    if not(os.path.exists(input_file)):
        print(f"Wav file is missing at {input_file}")
        return

    print(f"loading {input_file}")
    fs, clean_data = load_audio(input_file)
    noisy_data = add_white_noise(clean_data, noise_level = .02, seed=1)
    save_audio(noisy_out, fs, noisy_data)
    print(f"Noisy test file is made at {noisy_out}")

    # Decompose the signal
    wavelet_type = 'db20'
    decomposed_coeffs = pywt.wavedec(noisy_data, wavelet_type, level=9)
    
    # Denoise signal
    decomposed_coeffs = threshold(decomposed_coeffs, 0.035, 5)

    # Reconstruct signal
    denoised_data = pywt.waverec(decomposed_coeffs, wavelet_type)
    print(f"Noisy SNR = {snr(clean_data, noisy_data)} dB")
    print(f"Denoised SNR = {snr(clean_data, denoised_data)} dB")

    save_audio(processed_out, fs, denoised_data)
    print(f"Saved cleaned audio to {processed_out}")

if __name__ == "__main__":
    main()