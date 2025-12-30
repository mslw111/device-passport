import numpy as np
from scipy import signal

def run_fft_welch(samples, rate):
    if not samples: return {"peak_freq_hz": 0, "snr_db": 0, "total_power": 0}
    x = np.array(samples) # FIX
    nperseg = min(256, len(x))
    f, Pxx = signal.welch(x, fs=rate, nperseg=nperseg)
    peak_idx = np.argmax(Pxx)
    snr = 10 * np.log10(Pxx[peak_idx] / np.mean(Pxx)) if np.mean(Pxx) > 0 else 0
    return {
        "peak_freq_hz": float(f[peak_idx]),
        "total_power": float(np.sum(Pxx)),
        "snr_db": float(snr)
    }
def evaluate_r2v3_risk(d): return {}
