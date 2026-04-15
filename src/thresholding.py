import pywt
import numpy as np
#Determine which coefficients of waves stay and which are removed
def threshold(data, value, start=1, mode='soft',substitute=0):
    for i in range(start, len(data)):
        data[i] = pywt.threshold(data[i], value, mode, substitute)
    return data