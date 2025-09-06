
from __future__ import annotations
from typing import List, Tuple
from flask import Flask, request, jsonify
import numpy as np
import logging
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
    n = len(arr)

    # Step 1: baseline linear interpolation
    base = _linear_fill_with_edge_extrapolation(arr)

    # Step 2: fit global polynomial (deg 1 or 2)
    idx = np.where(~mask)[0]
    vals = arr[idx]
    deg = 1
    if len(idx) > 20:  # enough points
        # test if quadratic improves fit
        lin_coefs = np.polyfit(idx, vals, 1)
        quad_coefs = np.polyfit(idx, vals, 2)
        lin_pred = np.polyval(lin_coefs, idx)
        quad_pred = np.polyval(quad_coefs, idx)
        lin_err = np.mean((lin_pred - vals) ** 2)
        quad_err = np.mean((quad_pred - vals) ** 2)
        if quad_err < 0.95 * lin_err:
            deg = 2
            coefs = quad_coefs
        else:
            coefs = lin_coefs
    else:
        coefs = np.polyfit(idx, vals, 1)

    trend = np.polyval(coefs, np.arange(n))

    # Step 3: residuals
    residuals = arr - trend
    residuals[mask] = np.nan
    residuals_filled = _linear_fill_with_edge_extrapolation(residuals)
    residuals_smooth = _savgol_smooth(residuals_filled, window=31, poly=2)

    # Step 4: combine
    imputed = trend + residuals_smooth
    imputed = _clip_outliers_like(imputed, base)

    return imputed.tolist()


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
    series: List[List[float]] = payload["series"]
    logging.info("data sent for evaluation {}".format(payload))
    print("payload:", payload)
    answer = [impute_one(s) for s in series]

    return jsonify({"answer": answer}), 200