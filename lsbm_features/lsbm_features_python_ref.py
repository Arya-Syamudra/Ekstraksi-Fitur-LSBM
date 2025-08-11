# lsbm_features/lsbm_features_python_ref.py
"""
Python reference implementation of correlation features from
Liu et al., "Image Complexity and Feature Extraction for Steganalysis of LSB Matching Steganography" (2006).

Provides:
- features_54_python(img: np.ndarray) -> np.ndarray (54,)
- features_all_python(img: np.ndarray) -> np.ndarray (135,)

Notes:
- img: HxWx3 uint8 (BGR as loaded by OpenCV). We process channels as R,G,B in that order.
- Wavelet denoising: single-level Haar DWT via pywt.dwt2 / pywt.idwt2, detail coeffs with |coef| < t set to 0.
- Correlation computed as Pearson sample correlation with ddof=1; returns 0.0 if denom==0.
"""

from typing import List
import numpy as np
import cv2
import pywt

# ---------------- Utilities ----------------
def _pearson_corr(A: np.ndarray, B: np.ndarray) -> float:
    A = np.asarray(A, dtype=np.float64).ravel()
    B = np.asarray(B, dtype=np.float64).ravel()
    if A.size == 0 or B.size == 0 or A.size != B.size:
        return 0.0
    a = A - A.mean()
    b = B - B.mean()
    denom = (a.std(ddof=1) * b.std(ddof=1))
    if denom == 0 or np.isnan(denom):
        return 0.0
    cov = (a * b).sum() / (A.size - 1)
    r = cov / denom
    if np.isnan(r):
        return 0.0
    return float(r)

def _hist_prob(channel: np.ndarray) -> np.ndarray:
    hist = np.bincount(channel.ravel().astype(np.int64), minlength=256)[:256].astype(np.float64)
    s = hist.sum()
    if s == 0:
        return np.zeros(256, dtype=np.float64)
    return hist / s

def _autocorr_bitplane(M: np.ndarray, k: int, l: int) -> float:
    m, n = M.shape
    if k >= m or l >= n:
        return 0.0
    A = M[0:m-k, 0:n-l]
    B = M[k:m, l:n]
    return _pearson_corr(A, B)

# ---------------- Wavelet denoise ----------------
def _denoise_channel(channel: np.ndarray, t: float, wavelet: str = 'haar') -> np.ndarray:
    arr = np.asarray(channel, dtype=np.float64)
    # single-level 2D DWT
    LL, (LH, HL, HH) = pywt.dwt2(arr, wavelet, mode='periodization')
    # threshold detail coefficients
    LH = np.where(np.abs(LH) < t, 0.0, LH)
    HL = np.where(np.abs(HL) < t, 0.0, HL)
    HH = np.where(np.abs(HH) < t, 0.0, HH)
    rec = pywt.idwt2((LL, (LH, HL, HH)), wavelet, mode='periodization')
    rec = rec[:arr.shape[0], :arr.shape[1]]
    return rec.astype(np.float64)

def _CE(channel: np.ndarray, t: float, k: int, l: int) -> float:
    deno = _denoise_channel(channel, t)
    E = channel.astype(np.float64) - deno
    m, n = E.shape
    if k >= m or l >= n:
        return 0.0
    A = E[0:m-k, 0:n-l]
    B = E[k:m, l:n]
    return _pearson_corr(A, B)

# ---------------- Per-channel full features C1..C41 ----------------
def _features_channel_all(channel: np.ndarray) -> List[float]:
    ch = np.asarray(channel, dtype=np.uint8)
    M1 = ((ch >> 0) & 1).astype(np.uint8)
    M2 = ((ch >> 1) & 1).astype(np.uint8)
    feats: List[float] = []
    # C1
    feats.append(_pearson_corr(M1, M2))
    # C2..C15
    shifts = [(1,0),(2,0),(3,0),(4,0),
              (0,1),(0,2),(0,3),(0,4),
              (1,1),(2,2),(3,3),(4,4),
              (1,2),(2,1)]
    for (k,l) in shifts:
        feats.append(_autocorr_bitplane(M1, k, l))
    # C16: corr(He, Ho)
    rho = _hist_prob(ch)
    He = rho[0::2]
    Ho = rho[1::2]
    feats.append(_pearson_corr(He, Ho))
    # C17..C20: CH(l) for l=1..4
    for l in range(1,5):
        if rho.size <= l:
            feats.append(0.0)
        else:
            Hl1 = rho[:rho.size - l]
            Hl2 = rho[l:]
            feats.append(_pearson_corr(Hl1, Hl2))
    # C21..C41: CE for t in {1.5,2.0,2.5} and shifts
    thresholds = [1.5, 2.0, 2.5]
    shifts_t = [(0,1),(1,0),(1,1),(0,2),(2,0),(1,2),(2,1)]
    for t in thresholds:
        for (k,l) in shifts_t:
            feats.append(_CE(ch, t, k, l))
    # pad if necessary
    if len(feats) != 41:
        feats += [0.0] * (41 - len(feats))
    return feats

# ---------------- Subset for 54 features (14 per channel) ----------------
def _features_channel_subset_for_54(channel: np.ndarray) -> List[float]:
    ch = np.asarray(channel, dtype=np.uint8)
    feats: List[float] = []
    M1 = ((ch >> 0) & 1).astype(np.uint8)
    M2 = ((ch >> 1) & 1).astype(np.uint8)
    # C1
    feats.append(_pearson_corr(M1, M2))
    # C2 = (1,0)
    feats.append(_autocorr_bitplane(M1, 1, 0))
    # C6 = (0,1)
    feats.append(_autocorr_bitplane(M1, 0, 1))
    # C10 = (1,1)
    feats.append(_autocorr_bitplane(M1, 1, 1))
    # C14 = (1,2)
    feats.append(_autocorr_bitplane(M1, 1, 2))
    # C15 = (2,1)
    feats.append(_autocorr_bitplane(M1, 2, 1))
    # C16
    rho = _hist_prob(ch)
    He = rho[0::2]
    Ho = rho[1::2]
    feats.append(_pearson_corr(He, Ho))
    # C17 = CH(1)
    if rho.size > 1:
        Hl1 = rho[:rho.size-1]
        Hl2 = rho[1:]
        feats.append(_pearson_corr(Hl1, Hl2))
    else:
        feats.append(0.0)
    # CE(t; k,l) for t in {2.5,3.0}, shifts {(0,1),(1,0),(1,1)}
    for t in [2.5, 3.0]:
        for (k,l) in [(0,1),(1,0),(1,1)]:
            feats.append(_CE(ch, t, k, l))
    if len(feats) != 14:
        feats += [0.0] * (14 - len(feats))
    return feats

# ---------------- Cross-channel features ----------------
def _cross_channel_residuals(R: np.ndarray, G: np.ndarray, B: np.ndarray, thr_list: List[float]) -> List[float]:
    out: List[float] = []
    for t in thr_list:
        Er = R.astype(np.float64) - _denoise_channel(R, t)
        Eg = G.astype(np.float64) - _denoise_channel(G, t)
        Eb = B.astype(np.float64) - _denoise_channel(B, t)
        out.append(_pearson_corr(Er, Eg))
        out.append(_pearson_corr(Er, Eb))
        out.append(_pearson_corr(Eg, Eb))
    return out

def _lsb_cross_channel_abs(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> List[float]:
    Mr1 = ((R >> 0) & 1).astype(np.uint8)
    Mg1 = ((G >> 0) & 1).astype(np.uint8)
    Mb1 = ((B >> 0) & 1).astype(np.uint8)
    return [abs(_pearson_corr(Mr1, Mg1)),
            abs(_pearson_corr(Mr1, Mb1)),
            abs(_pearson_corr(Mg1, Mb1))]

# ---------------- Public API ----------------
def features_all_python(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("img is None")
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("img must be HxWx3 color image")
    b, g, r = cv2.split(arr)   # OpenCV loads BGR; we process R,G,B order as in paper mapping
    feats: List[float] = []
    feats.extend(_features_channel_all(r))
    feats.extend(_features_channel_all(g))
    feats.extend(_features_channel_all(b))
    feats.extend(_cross_channel_residuals(r, g, b, [1.0, 1.5, 2.0]))  # 9
    feats.extend(_lsb_cross_channel_abs(r, g, b))  # 3
    if len(feats) != 135:
        if len(feats) < 135:
            feats += [0.0] * (135 - len(feats))
        else:
            feats = feats[:135]
    return np.asarray(feats, dtype=np.float64)

def features_54_python(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("img is None")
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("img must be HxWx3 color image")
    b, g, r = cv2.split(arr)
    feats: List[float] = []
    feats.extend(_features_channel_subset_for_54(r))
    feats.extend(_features_channel_subset_for_54(g))
    feats.extend(_features_channel_subset_for_54(b))
    feats.extend(_cross_channel_residuals(r, g, b, [1.0, 1.5, 2.0]))  # 9
    feats.extend(_lsb_cross_channel_abs(r, g, b))  # 3
    if len(feats) != 54:
        if len(feats) < 54:
            feats += [0.0] * (54 - len(feats))
        else:
            feats = feats[:54]
    return np.asarray(feats, dtype=np.float64)

# path-based wrappers
def features_all_python_from_path(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return features_all_python(img)

def features_54_python_from_path(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return features_54_python(img)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        p = sys.argv[1]
        a54 = features_54_python_from_path(p)
        a135 = features_all_python_from_path(p)
        print("54 len:", a54.shape)
        print("135 len:", a135.shape)
    else:
        print("Usage: python lsbm_features_python_ref.py /path/to/image")
