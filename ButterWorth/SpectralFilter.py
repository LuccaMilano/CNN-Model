import numpy as np
import scipy.signal as signal

def apply_bandpass_filter(data, b, a):
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data

def design_butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a