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
    wavelet_type = 'db8'
    level = 5
    decomposed_coeffs = pywt.wavedec(noisy_data, wavelet_type, level=level)

    denoised_coeffs = [decomposed_coeffs[0]]
    n = len(noisy_data)
    for i in range(1, len(decomposed_coeffs)):
        level_sigma = np.median(np.abs(decomposed_coeffs[i])) / .6745
        threshold_value = level_sigma * np.sqrt(2 * np.log(n))
        scale_factor = .15 + (i / len(decomposed_coeffs)) * .35
        applied_thresh = threshold_value * scale_factor
        denoised_coeffs.append(pywt.threshold(decomposed_coeffs[i], value=applied_thresh))
    
    denoised_data = pywt.waverec(denoised_coeffs, wavelet_type)

    if len(denoised_data) > len(clean_data):
        denoised_data = denoised_data[:len(clean_data)]

    print(f"Noisy SNR = {snr(clean_data, noisy_data):.2f} dB")
    print(f"Denoised SNR = {snr(clean_data, denoised_data):.2f} dB")

    save_audio(processed_out, fs, denoised_data)

    '''sigma = np.median(np.abs(decomposed_coeffs[-1])) / .6745
    threshold_value  = sigma * np.sqrt(2 * np.log(len(noisy_data)))
    denoised_coeffs = [decomposed_coeffs[0]]
    for i in range(1, len(decomposed_coeffs)):
        denoised_coeffs.append(pywt.threshold(decomposed_coeffs[i], value= (threshold_value * .2), mode = 'soft'))
    
    # Denoise signal
    #decomposed_coeffs = threshold(decomposed_coeffs, threshold_value, start = 1)

    # Reconstruct signal
    denoised_data = pywt.waverec(denoised_coeffs, wavelet_type)
    print(f"Noisy SNR = {snr(clean_data, noisy_data)} dB")
    print(f"Denoised SNR = {snr(clean_data, denoised_data)} dB")

    save_audio(processed_out, fs, denoised_data)
    print(f"Saved cleaned audio to {processed_out}")'''



    #Plots and Graphs
    plot_dir = "graphsNMore"
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(12,10))

    plt.subplot(2, 1, 1)
    plt.plot(clean_data[10000:15000], label = "Clean", alpha = .7)
    plt.plot(noisy_data[10000:15000], label = "Noisy", alpha = .5)
    plt.title("Time: Clean vs Noisy")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(clean_data[10000:15000], label = "Clean Ground Truth", color = "gray", alpha = .3)
    plt.plot(denoised_data[10000:15000], label = "Denoised Wavelet", color = "orange")
    plt.title(f"Denoised Result (SNR : {snr(clean_data, denoised_data): .2f} dB)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/waveform_comparison.png")
    #plt.show()

    plt.figure(figsize=(10,6))
    plt.specgram(denoised_data, Fs = fs)
    plt.title("Spectrogram of Denoised Audio")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.savefig(f"{plot_dir}/spectrogram_denoised.png")

if __name__ == "__main__":
    main()