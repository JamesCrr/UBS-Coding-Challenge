from __future__ import annotations
from typing import List, Tuple
from flask import Flask, request, jsonify
import numpy as np
import math

from flask import request
from routes import app

def _linear_fill_with_edge_extrapolation(x: np.ndarray) -> np.ndarray:
    """
    Fill NaNs by linear interpolation on observed points.
    Extrapolates linearly at the edges using the first/last two observed points.
    If fewer than 2 observed points exist, falls back to constant fill (single observed value or zeros).
    """
    n = x.size
    y = x.copy()
    isnan = np.isnan(y)
    if np.all(isnan):
        return np.zeros_like(y)

    idx = np.where(~isnan)[0]
    vals = y[idx]

    # If only one observed value: constant fill
    if idx.size == 1:
        y[:] = vals[0]
        return y

    # Build full set of x positions
    pos = np.arange(n)

    # Internal interpolation
    y[isnan] = np.interp(pos[isnan], idx, vals)

    # For completeness, np.interp already handles edge extrapolation
    # by using the first/last observed values, which is constant extrapolation.
    # Upgrade to linear edge extrapolation when possible:
    first_obs, second_obs = idx[0], idx[1]
    last_obs, penult_obs = idx[-1], idx[-2]

    # Left edge linear extrapolation (if gap before first_obs)
    if first_obs > 0:
        slope_left = (y[second_obs] - y[first_obs]) / (second_obs - first_obs)
        for k in range(first_obs - 1, -1, -1):
            y[k] = y[k + 1] - slope_left

    # Right edge linear extrapolation (if gap after last_obs)
    if last_obs < n - 1:
        slope_right = (y[last_obs] - y[penult_obs]) / (last_obs - penult_obs)
        for k in range(last_obs + 1, n):
            y[k] = y[k - 1] + slope_right

    return y


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """
    Two-sided moving average with reflective padding.
    """
    if window <= 1:
        return signal.copy()
    w = int(window)
    pad = w // 2
    # Reflective padding to avoid edge shrink
    padded = np.pad(signal, (pad, pad), mode="reflect")
    kernel = np.ones(w, dtype=float) / w
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _adaptive_blend(original: np.ndarray,
                    smoothed: np.ndarray,
                    miss_mask: np.ndarray,
                    w_missing: float = 0.7,
                    w_observed: float = 0.2) -> np.ndarray:
    """
    Blend original and smoothed values with stronger smoothing on originally-missing indices.
    """
    w = np.where(miss_mask, w_missing, w_observed)
    return (1.0 - w) * original + w * smoothed


def _local_poly_refine(series: np.ndarray,
                       miss_mask: np.ndarray,
                       max_neighbors: int = 30) -> np.ndarray:
    """
    For each contiguous run of missing values, fit a local quadratic polynomial
    to up to 'max_neighbors' observed points on each side and refine estimates.
    Uses simple ridge (L2) regularization to ensure stability.
    """
    y = series.copy()
    n = y.size
    i = 0

    while i < n:
        if not miss_mask[i]:
            i += 1
            continue

        # Identify contiguous missing block [i, j)
        j = i
        while j < n and miss_mask[j]:
            j += 1

        # Gather neighborhood indices
        left_idx = np.arange(max(i - max_neighbors, 0), i)
        right_idx = np.arange(j, min(j + max_neighbors, n))
        nb_idx = np.concatenate([left_idx, right_idx])

        # If no neighbors, skip (shouldn't happen after first-stage fill)
        if nb_idx.size == 0:
            i = j
            continue

        x_nb = nb_idx.astype(float)
        y_nb = y[nb_idx].astype(float)

        # Center x to improve conditioning
        x0 = (i + j - 1) / 2.0
        x_c = x_nb - x0

        # Design matrix for quadratic: [1, x, x^2]
        X = np.column_stack([np.ones_like(x_c), x_c, x_c**2])

        # Ridge regularization
        lam = 1e-3
        XtX = X.T @ X + lam * np.eye(3)
        Xty = X.T @ y_nb
        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(X, y_nb, rcond=None)[0]

        # Predict for the missing block
        miss_idx = np.arange(i, j)
        x_m = miss_idx.astype(float) - x0
        X_m = np.column_stack([np.ones_like(x_m), x_m, x_m**2])
        y_hat = X_m @ beta

        # Blend with current estimate for stability
        y[miss_idx] = 0.5 * y[miss_idx] + 0.5 * y_hat

        i = j

    return y


def _clip_outliers_like(signal: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Clip 'signal' to a sane range derived from 'ref' distribution (1st..99th percentiles, extended by 3*IQR).
    Guards against extreme excursions after extrapolation.
    """
    finite_ref = ref[np.isfinite(ref)]
    if finite_ref.size == 0:
        return np.nan_to_num(signal, nan=0.0, posinf=1e6, neginf=-1e6)

    p1, p99 = np.percentile(finite_ref, [1, 99])
    q1, q3 = np.percentile(finite_ref, [25, 75])
    iqr = q3 - q1
    lo = p1 - 3.0 * iqr
    hi = p99 + 3.0 * iqr
    clipped = np.clip(signal, lo, hi)
    return np.nan_to_num(clipped, nan=0.0, posinf=hi, neginf=lo)


def impute_one(series_list: List[float]) -> List[float]:
    """
    Impute a single 1D series (length 1000) with possible nulls.
    Returns a Python list of floats (no NaNs/Infs).
    """
    # Convert None -> NaN
    arr = np.array([np.nan if (v is None) else float(v) for v in series_list], dtype=float)
    original_missing = np.isnan(arr)

    # Step 1: Linear interpolation + edge linear extrapolation
    filled = _linear_fill_with_edge_extrapolation(arr)

    # Step 2: Gentle denoising with adaptive smoothing
    n = filled.size
    # Window ~ 2%–5% of length: choose odd number
    win = int(max(5, min(51, (n // 20) | 1)))  # ensure odd-ish, but even is fine for average
    smoothed = _moving_average(filled, win)
    blended = _adaptive_blend(filled, smoothed, original_missing, w_missing=0.7, w_observed=0.2)

    # Step 3: Local quadratic refinement on missing runs
    refined = _local_poly_refine(blended, original_missing, max_neighbors=30)

    # Step 4: Final light smoothing pass on only the imputed points (preserve observed)
    smoothed2 = _moving_average(refined, window=max(5, win))
    refined = np.where(original_missing,
                       0.5 * refined + 0.5 * smoothed2,
                       refined)

    # Step 5: Safety clamps
    refined = _clip_outliers_like(refined, ref=filled)

    # Ensure finite numeric output
    refined = np.nan_to_num(refined, nan=0.0, posinf=np.finfo(float).max/2, neginf=-np.finfo(float).max/2)

    return refined.tolist()


def validate_payload(payload) -> Tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "Payload must be a JSON object"
    if "series" not in payload:
        return False, "Missing 'series' key"
    series = payload["series"]
    if not isinstance(series, list) or len(series) != 100:
        return False, "Expected 'series' to be a list of exactly 100 lists"
    for idx, row in enumerate(series):
        if not isinstance(row, list) or len(row) != 1000:
            return False, f"Each inner list must have exactly 1000 elements (problem at index {idx})"
        # Basic type check: allow float, int, or None
        for j, v in enumerate(row):
            if v is None:
                continue
            if not isinstance(v, (int, float)):
                return False, f"Element at series[{idx}][{j}] must be float, int, or null"
    return True, ""


@app.route('/blankety', methods=['POST'])
def blankety():
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid or missing JSON"}), 400

    ok, msg = validate_payload(payload)
    if not ok:
        return jsonify({"error": msg}), 400

    series: List[List[float]] = payload["series"]

    # Impute each list
    try:
        answer = [impute_one(s) for s in series]
    except Exception as e:
        return jsonify({"error": f"Imputation failed: {str(e)}"}), 500

    # Final shape & numeric checks
    if len(answer) != 100 or any(len(row) != 1000 for row in answer):
        return jsonify({"error": "Output shape mismatch"}), 500
    # Ensure numeric-only (no NaNs/Infs)
    for i, row in enumerate(answer):
        arr = np.array(row, dtype=float)
        if not np.all(np.isfinite(arr)):
            return jsonify({"error": f"Non-finite values produced in series {i}"}), 500

    return jsonify({"answer": answer}), 200