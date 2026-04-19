#Core for Discrete Wavelet Transform
#Also where MRA (Multiresolution Analysis should be at)
import numpy as np
import pywt

def dwt2d_mra(image, wavelet='db4', level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    return coeffs

def idwt2d_mra(coeffs, wavelet='db4'):
    return pywt.waverec2(coeffs, wavelet)

def apply_threshold(coeffs, value, mode='soft'):
    new_coeffs = [coeffs[0]]
    for detail_level in coeffs[1:]:
        thresholded_details = tuple(
            pywt.threshold(d, value, mode=mode) for d in detail_level
        )
        new_coeffs.append(thresholded_details)
    return new_coeffs