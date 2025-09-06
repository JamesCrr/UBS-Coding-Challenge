
from __future__ import annotations
from typing import List, Tuple
from flask import Flask, request, jsonify
import numpy as np
import logging
from scipy.signal import savgol_filter
from numpy.linalg import LinAlgError

from flask import request
from routes import app

# # -----------------------------
# # Imputation utilities
# # -----------------------------

# def _fit_poly(idx, vals, max_deg=2):
#     """Fit degree 1 or 2 polynomial safely with normalization."""
#     if len(idx) < 5:
#         # too few points, fallback
#         return np.polyfit(idx, vals, 1), 1

#     # normalize x to [-1,1]
#     x_min, x_max = idx.min(), idx.max()
#     x_scaled = (idx - x_min) / (x_max - x_min) * 2 - 1

#     # normalize y to mean ~0, std ~1
#     y_mean, y_std = np.mean(vals), np.std(vals) or 1.0
#     y_scaled = (vals - y_mean) / y_std

#     best_deg, best_err, best_coefs = 1, float("inf"), None
#     for deg in range(1, max_deg + 1):
#         try:
#             coefs = np.polyfit(x_scaled, y_scaled, deg)
#             pred = np.polyval(coefs, x_scaled)
#             # safer metric: MAE instead of MSE
#             err = np.nanmean(np.abs(pred - y_scaled))
#             if err < best_err:
#                 best_err = err
#                 best_deg = deg
#                 best_coefs = coefs
#         except Exception:
#             continue

#     # return normalized fit info
#     return (best_coefs, best_deg, x_min, x_max, y_mean, y_std)


# def _poly_predict(n, fit_info):
#     coefs, deg, x_min, x_max, y_mean, y_std = fit_info
#     x_full = (np.arange(n) - x_min) / (x_max - x_min) * 2 - 1
#     y_pred = np.polyval(coefs, x_full)
#     return y_pred * y_std + y_mean


# def _linear_fill_with_edge_extrapolation(x: np.ndarray) -> np.ndarray:
#     """Linear interpolate + edge linear extrapolation"""
#     n = x.size
#     y = x.copy()
#     isnan = np.isnan(y)
#     if np.all(isnan):
#         return np.zeros_like(y)

#     idx = np.where(~isnan)[0]
#     vals = y[idx]

#     if idx.size == 1:
#         y[:] = vals[0]
#         return y

#     pos = np.arange(n)
#     y[isnan] = np.interp(pos[isnan], idx, vals)

#     # Linear edge extrapolation
#     if idx[0] > 0:
#         slope = (vals[1] - vals[0]) / (idx[1] - idx[0])
#         for k in range(idx[0] - 1, -1, -1):
#             y[k] = y[k + 1] - slope
#     if idx[-1] < n - 1:
#         slope = (vals[-1] - vals[-2]) / (idx[-1] - idx[-2])
#         for k in range(idx[-1] + 1, n):
#             y[k] = y[k - 1] + slope

#     return y


# def _savgol_smooth(arr: np.ndarray, window: int = 51, poly: int = 3) -> np.ndarray:
#     """Savitzky–Golay smoothing with edge handling"""
#     if window >= len(arr):
#         window = len(arr) - (1 - len(arr) % 2)  # force odd < len
#     if window < 5:
#         return arr
#     return savgol_filter(arr, window_length=window, polyorder=poly, mode="mirror")


# def _yule_walker_estimate(x: np.ndarray, order: int = 3) -> np.ndarray:
#     """Estimate AR coefficients via Yule–Walker"""
#     x = x - np.mean(x)
#     r = np.correlate(x, x, mode="full")[len(x)-1:]
#     R = np.array([r[i:i+order] for i in range(order)])
#     rhs = r[1:order+1]
#     try:
#         phi = np.linalg.solve(R, rhs)
#     except LinAlgError:
#         phi = np.zeros(order)
#     return phi


# def _ar_predict(arr: np.ndarray, idx_missing: np.ndarray, order: int = 3, neigh: int = 50) -> np.ndarray:
#     """Predict missing points using AR model fitted on neighbors"""
#     y = arr.copy()
#     n = len(arr)
#     for i in idx_missing:
#         left = max(0, i - neigh)
#         right = min(n, i + neigh)
#         segment = np.delete(y[left:right], np.where(np.isnan(y[left:right])))
#         if len(segment) < order + 1:
#             continue
#         phi = _yule_walker_estimate(segment, order=order)
#         # AR prediction using last known points
#         history = []
#         for k in range(order):
#             pos = i - (k + 1)
#             if pos >= 0:
#                 history.append(y[pos])
#         if len(history) < order:
#             continue
#         pred = np.dot(phi, history[::-1])
#         y[i] = pred
#     return y


# def _clip_outliers_like(signal: np.ndarray, ref: np.ndarray) -> np.ndarray:
#     """Clamp values based on distribution of reference"""
#     finite_ref = ref[np.isfinite(ref)]
#     if finite_ref.size == 0:
#         return np.nan_to_num(signal)
#     p1, p99 = np.percentile(finite_ref, [1, 99])
#     q1, q3 = np.percentile(finite_ref, [25, 75])
#     iqr = q3 - q1
#     lo = p1 - 3.0 * iqr
#     hi = p99 + 3.0 * iqr
#     clipped = np.clip(signal, lo, hi)
#     return np.nan_to_num(clipped)


# def impute_one(series_list: List[float]) -> List[float]:
#     arr = np.array([np.nan if (v is None) else float(v) for v in series_list], dtype=float)
#     mask = np.isnan(arr)
#     n = len(arr)

#     # Step 1: baseline linear interpolation
#     base = _linear_fill_with_edge_extrapolation(arr)

#     # Step 2: fit global polynomial (deg 1 or 2)
#     idx = np.where(~mask)[0]
#     vals = arr[idx]
#     deg = 1
#     if len(idx) > 20:  # enough points
#         # test if quadratic improves fit
#         lin_coefs = np.polyfit(idx, vals, 1)
#         quad_coefs = np.polyfit(idx, vals, 2)
#         lin_pred = np.polyval(lin_coefs, idx)
#         quad_pred = np.polyval(quad_coefs, idx)
#         lin_err = np.mean((lin_pred - vals) ** 2)
#         quad_err = np.mean((quad_pred - vals) ** 2)
#         if quad_err < 0.95 * lin_err:
#             deg = 2
#             coefs = quad_coefs
#         else:
#             coefs = lin_coefs
#     else:
#         coefs = np.polyfit(idx, vals, 1)

#     # trend = np.polyval(coefs, np.arange(n))

#     # coefs, deg, xmin, scale = _fit_poly(idx, vals, max_deg=2)
#     # trend = _poly_predict(n, coefs, deg, xmin, scale)

#     fit_info = _fit_poly(idx, vals, max_deg=2)
#     trend = _poly_predict(n, fit_info)

#     # Step 3: residuals
#     residuals = arr - trend
#     residuals[mask] = np.nan
#     residuals_filled = _linear_fill_with_edge_extrapolation(residuals)
#     residuals_smooth = _savgol_smooth(residuals_filled, window=31, poly=2)

#     # Step 4: combine
#     imputed = trend + residuals_smooth
#     imputed = _clip_outliers_like(imputed, base)

#     return imputed.tolist()


# def validate_payload(payload) -> Tuple[bool, str]:
#     if not isinstance(payload, dict) or "series" not in payload:
#         return False, "Missing 'series' key"
#     series = payload["series"]
#     if not isinstance(series, list) or len(series) != 100:
#         return False, "Expected 'series' to be a list of exactly 100 lists"
#     for row in series:
#         if not isinstance(row, list) or len(row) != 1000:
#             return False, "Each list must have exactly 1000 elements"
#     return True, ""


# @app.route('/blankety', methods=['POST'])
# def blankety():
#     payload = request.get_json(force=True, silent=False)
#     ok, msg = validate_payload(payload)
#     if not ok:
#         return jsonify({"error": msg}), 400
#     series: List[List[float]] = payload["series"]
#     logging.info("data sent for evaluation {}".format(payload))
#     print("payload:", payload)
#     answer = [impute_one(s) for s in series]

#     return jsonify({"answer": answer}), 200






















def robust_impute_series(series):
    """
    Robust imputation for a single time series using multiple techniques.
    """
    series = np.array(series, dtype=float)
    n = len(series)
    
    # Find valid (non-null) indices
    valid_mask = ~np.isnan(series)
    valid_indices = np.where(valid_mask)[0]
    valid_values = series[valid_mask]
    
    # If no missing values, return as is
    if len(valid_values) == n:
        return series.tolist()
    
    # If too few valid points, use simple interpolation
    if len(valid_values) < 3:
        return simple_interpolate(series).tolist()
    
    # Create index array for interpolation
    all_indices = np.arange(n)
    
    # Method 1: Cubic spline interpolation (primary method)
    try:
        # Use cubic spline with smoothing factor
        smoothing_factor = len(valid_values) * 0.1  # Adaptive smoothing
        spline = interpolate.UnivariateSpline(valid_indices, valid_values, 
                                            s=smoothing_factor, k=min(3, len(valid_values)-1))
        spline_result = spline(all_indices)
        
        # Clamp to reasonable bounds based on data range
        data_min, data_max = np.min(valid_values), np.max(valid_values)
        data_range = data_max - data_min
        lower_bound = data_min - 0.5 * data_range
        upper_bound = data_max + 0.5 * data_range
        spline_result = np.clip(spline_result, lower_bound, upper_bound)
        
    except:
        spline_result = simple_interpolate(series)
    
    # Method 2: Local regression (LOWESS-like approach)
    try:
        lowess_result = local_regression_impute(series, valid_indices, valid_values)
    except:
        lowess_result = spline_result.copy()
    
    # Method 3: Trend + residual decomposition for long sequences
    try:
        if len(valid_values) > 20:
            trend_result = trend_based_impute(series, valid_indices, valid_values)
        else:
            trend_result = spline_result.copy()
    except:
        trend_result = spline_result.copy()
    
    # Ensemble: Weight different methods based on data characteristics
    result = ensemble_impute(series, spline_result, lowess_result, trend_result, valid_mask)
    
    # Final safety checks
    result = np.nan_to_num(result, nan=np.nanmean(valid_values))
    result = np.where(np.isinf(result), np.nanmean(valid_values), result)
    
    return result.tolist()

def simple_interpolate(series):
    """Simple linear interpolation fallback."""
    series = np.array(series, dtype=float)
    valid_mask = ~np.isnan(series)
    
    if np.sum(valid_mask) == 0:
        return np.zeros_like(series)
    
    valid_indices = np.where(valid_mask)[0]
    valid_values = series[valid_mask]
    
    # Linear interpolation
    result = np.interp(np.arange(len(series)), valid_indices, valid_values)
    return result

def local_regression_impute(series, valid_indices, valid_values):
    """Local regression imputation."""
    n = len(series)
    result = np.full(n, np.nan)
    result[valid_indices] = valid_values
    
    # For each missing point, use local weighted regression
    missing_indices = np.where(np.isnan(series))[0]
    
    for idx in missing_indices:
        # Find nearest neighbors
        distances = np.abs(valid_indices - idx)
        # Use adaptive window size
        window_size = min(max(10, len(valid_indices) // 10), len(valid_indices))
        nearest_indices = valid_indices[np.argsort(distances)[:window_size]]
        nearest_values = series[nearest_indices]
        
        # Weight by distance (tricube kernel)
        weights = distances[np.argsort(distances)[:window_size]]
        max_dist = np.max(weights) + 1e-8
        weights = (1 - (weights / max_dist) ** 3) ** 3
        
        # Weighted linear fit
        try:
            coeffs = np.polyfit(nearest_indices, nearest_values, deg=1, w=weights)
            result[idx] = np.polyval(coeffs, idx)
        except:
            # Fallback to weighted average
            result[idx] = np.average(nearest_values, weights=weights)
    
    # Fill remaining with linear interpolation
    still_missing = np.isnan(result)
    if np.any(still_missing):
        valid_now = ~still_missing
        result[still_missing] = np.interp(np.where(still_missing)[0], 
                                        np.where(valid_now)[0], 
                                        result[valid_now])
    
    return result

def trend_based_impute(series, valid_indices, valid_values):
    """Trend-based imputation with residual modeling."""
    n = len(series)
    
    # Fit polynomial trend
    try:
        # Use adaptive degree based on data length
        degree = min(3, len(valid_values) // 10)
        degree = max(1, degree)
        
        trend_coeffs = np.polyfit(valid_indices, valid_values, degree)
        trend_all = np.polyval(trend_coeffs, np.arange(n))
        
        # Calculate residuals at valid points
        trend_valid = np.polyval(trend_coeffs, valid_indices)
        residuals = valid_values - trend_valid
        
        # Smooth residuals with moving average
        if len(residuals) > 5:
            # Create smooth residual function
            residual_spline = interpolate.interp1d(valid_indices, residuals, 
                                                 kind='linear', 
                                                 bounds_error=False, 
                                                 fill_value=0)
            residuals_all = residual_spline(np.arange(n))
        else:
            residuals_all = np.zeros(n)
        
        result = trend_all + residuals_all
        
    except:
        # Fallback to simple polynomial fit
        coeffs = np.polyfit(valid_indices, valid_values, min(2, len(valid_values)-1))
        result = np.polyval(coeffs, np.arange(n))
    
    return result

def ensemble_impute(series, spline_result, lowess_result, trend_result, valid_mask):
    """Ensemble different imputation methods."""
    # Calculate weights based on local data density and variability
    n = len(series)
    result = np.full(n, np.nan)
    
    # Keep original values
    result[valid_mask] = series[valid_mask]
    
    # For missing values, use weighted ensemble
    missing_mask = ~valid_mask
    missing_indices = np.where(missing_mask)[0]
    
    for idx in missing_indices:
        # Local data characteristics around this point
        window = 20  # Look at ±20 points around missing value
        start_idx = max(0, idx - window)
        end_idx = min(n, idx + window + 1)
        
        local_valid = valid_mask[start_idx:end_idx]
        local_density = np.sum(local_valid) / len(local_valid)
        
        # Weight methods based on local characteristics
        if local_density > 0.7:  # High density - trust spline more
            weights = [0.6, 0.3, 0.1]  # spline, lowess, trend
        elif local_density > 0.3:  # Medium density - balanced
            weights = [0.4, 0.4, 0.2]
        else:  # Low density - trust trend more
            weights = [0.3, 0.2, 0.5]
        
        # Ensemble prediction
        predictions = [spline_result[idx], lowess_result[idx], trend_result[idx]]
        
        # Remove any invalid predictions
        valid_predictions = []
        valid_weights = []
        for pred, weight in zip(predictions, weights):
            if np.isfinite(pred):
                valid_predictions.append(pred)
                valid_weights.append(weight)
        
        if valid_predictions:
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / np.sum(valid_weights)  # Normalize
            result[idx] = np.average(valid_predictions, weights=valid_weights)
        else:
            # Ultimate fallback
            result[idx] = np.nanmean(series[valid_mask])
    
    return result

@app.route('/blankety', methods=['POST'])
def blankety():
    try:
        data = request.get_json()
        series_list = data['series']
        
        # Process each series
        imputed_series = []
        for series in series_list:
            # Convert None to np.nan for processing
            series_array = [np.nan if x is None else x for x in series]
            imputed = robust_impute_series(series_array)
            imputed_series.append(imputed)
        
        return jsonify({"answer": imputed_series})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500