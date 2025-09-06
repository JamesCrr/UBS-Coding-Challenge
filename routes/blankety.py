
from __future__ import annotations
from typing import List, Tuple
from flask import Flask, request, jsonify
import numpy as np
from scipy.signal import savgol_filter
from numpy.linalg import LinAlgError

from flask import request
from routes import app

# -----------------------------
# Imputation utilities
# -----------------------------

def _linear_fill_with_edge_extrapolation(x: np.ndarray) -> np.ndarray:
    """Linear interpolate + edge linear extrapolation"""
    n = x.size
    y = x.copy()
    isnan = np.isnan(y)
    if np.all(isnan):
        return np.zeros_like(y)

    idx = np.where(~isnan)[0]
    vals = y[idx]

    if idx.size == 1:
        y[:] = vals[0]
        return y

    pos = np.arange(n)
    y[isnan] = np.interp(pos[isnan], idx, vals)

    # Linear edge extrapolation
    if idx[0] > 0:
        slope = (vals[1] - vals[0]) / (idx[1] - idx[0])
        for k in range(idx[0] - 1, -1, -1):
            y[k] = y[k + 1] - slope
    if idx[-1] < n - 1:
        slope = (vals[-1] - vals[-2]) / (idx[-1] - idx[-2])
        for k in range(idx[-1] + 1, n):
            y[k] = y[k - 1] + slope

    return y


def _savgol_smooth(arr: np.ndarray, window: int = 51, poly: int = 3) -> np.ndarray:
    """Savitzky–Golay smoothing with edge handling"""
    if window >= len(arr):
        window = len(arr) - (1 - len(arr) % 2)  # force odd < len
    if window < 5:
        return arr
    return savgol_filter(arr, window_length=window, polyorder=poly, mode="mirror")


def _yule_walker_estimate(x: np.ndarray, order: int = 3) -> np.ndarray:
    """Estimate AR coefficients via Yule–Walker"""
    x = x - np.mean(x)
    r = np.correlate(x, x, mode="full")[len(x)-1:]
    R = np.array([r[i:i+order] for i in range(order)])
    rhs = r[1:order+1]
    try:
        phi = np.linalg.solve(R, rhs)
    except LinAlgError:
        phi = np.zeros(order)
    return phi


def _ar_predict(arr: np.ndarray, idx_missing: np.ndarray, order: int = 3, neigh: int = 50) -> np.ndarray:
    """Predict missing points using AR model fitted on neighbors"""
    y = arr.copy()
    n = len(arr)
    for i in idx_missing:
        left = max(0, i - neigh)
        right = min(n, i + neigh)
        segment = np.delete(y[left:right], np.where(np.isnan(y[left:right])))
        if len(segment) < order + 1:
            continue
        phi = _yule_walker_estimate(segment, order=order)
        # AR prediction using last known points
        history = []
        for k in range(order):
            pos = i - (k + 1)
            if pos >= 0:
                history.append(y[pos])
        if len(history) < order:
            continue
        pred = np.dot(phi, history[::-1])
        y[i] = pred
    return y


def _clip_outliers_like(signal: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Clamp values based on distribution of reference"""
    finite_ref = ref[np.isfinite(ref)]
    if finite_ref.size == 0:
        return np.nan_to_num(signal)
    p1, p99 = np.percentile(finite_ref, [1, 99])
    q1, q3 = np.percentile(finite_ref, [25, 75])
    iqr = q3 - q1
    lo = p1 - 3.0 * iqr
    hi = p99 + 3.0 * iqr
    clipped = np.clip(signal, lo, hi)
    return np.nan_to_num(clipped)


def impute_one(series_list: List[float]) -> List[float]:
    arr = np.array([np.nan if (v is None) else float(v) for v in series_list], dtype=float)
    mask = np.isnan(arr)

    # Step 1: Linear fill
    base = _linear_fill_with_edge_extrapolation(arr)

    # Step 2: Savitzky–Golay smoothing
    smooth = _savgol_smooth(base, window=51, poly=3)

    # Step 3: AR refinement
    ar_filled = _ar_predict(base.copy(), np.where(mask)[0], order=3, neigh=50)

    # Step 4: Blend
    blended = np.where(mask, 0.6 * ar_filled + 0.4 * smooth, base)

    # Step 5: Clamp outliers
    final = _clip_outliers_like(blended, base)

    return final.tolist()


def validate_payload(payload) -> Tuple[bool, str]:
    if not isinstance(payload, dict) or "series" not in payload:
        return False, "Missing 'series' key"
    series = payload["series"]
    if not isinstance(series, list) or len(series) != 100:
        return False, "Expected 'series' to be a list of exactly 100 lists"
    for row in series:
        if not isinstance(row, list) or len(row) != 1000:
            return False, "Each list must have exactly 1000 elements"
    return True, ""


@app.route('/blankety', methods=['POST'])
def blankety():
    payload = request.get_json(force=True, silent=False)
    ok, msg = validate_payload(payload)
    if not ok:
        return jsonify({"error": msg}), 400
    print("msg:", msg)
    series: List[List[float]] = payload["series"]
    answer = [impute_one(s) for s in series]

    return jsonify({"answer": answer}), 200