from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


def _clip_crop_bounds(image: np.ndarray, box: Dict[str, Any]) -> tuple[int, int, int, int]:
    image_height, image_width = image.shape[:2]
    left = max(int(round(float(box.get("x", 0.0)))), 0)
    top = max(int(round(float(box.get("y", 0.0)))), 0)
    right = min(int(round(float(box.get("right", left)))), image_width)
    bottom = min(int(round(float(box.get("bottom", top)))), image_height)
    return left, top, right, bottom


def _estimate_background_bgr(crop: np.ndarray) -> np.ndarray:
    height, width = crop.shape[:2]
    band_y = max(1, int(round(height * 0.08)))
    band_x = max(1, int(round(width * 0.08)))
    border_pixels = np.concatenate(
        [
            crop[:band_y, :, :].reshape(-1, 3),
            crop[-band_y:, :, :].reshape(-1, 3),
            crop[:, :band_x, :].reshape(-1, 3),
            crop[:, -band_x:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(border_pixels, axis=0)


def _smooth_profile(profile: np.ndarray, window: int) -> np.ndarray:
    if profile.size == 0:
        return profile
    window = max(1, window)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(profile, kernel, mode="same")


def _largest_inactive_run(active_rows: np.ndarray) -> int:
    best = 0
    current = 0
    for is_active in active_rows.tolist():
        if is_active:
            best = max(best, current)
            current = 0
        else:
            current += 1
    return max(best, current)


def _coarse_grid_occupancy_ratio(
    active_bool: np.ndarray,
    *,
    target_rows: int,
    target_cols: int,
    min_cell_fill_ratio: float,
) -> float:
    if active_bool.size == 0:
        return 0.0

    height, width = active_bool.shape
    rows = max(1, min(target_rows, height))
    cols = max(1, min(target_cols, width))

    occupied_cells = 0
    total_cells = rows * cols
    for row_index in range(rows):
        top = int(round(row_index * height / rows))
        bottom = int(round((row_index + 1) * height / rows))
        if bottom <= top:
            bottom = min(top + 1, height)
        for col_index in range(cols):
            left = int(round(col_index * width / cols))
            right = int(round((col_index + 1) * width / cols))
            if right <= left:
                right = min(left + 1, width)
            cell = active_bool[top:bottom, left:right]
            if cell.size == 0:
                continue
            if float(cell.mean()) >= min_cell_fill_ratio:
                occupied_cells += 1
    return float(occupied_cells) / float(total_cells)


def compute_visual_density_metrics(
    image_path: str | Path,
    container_box: Dict[str, Any],
    *,
    lower_band_start_ratio: float = 0.55,
) -> Dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load rendered image: {image_path}")

    left, top, right, bottom = _clip_crop_bounds(image, container_box)
    if right <= left or bottom <= top:
        raise ValueError("Container crop is empty after clipping")

    crop = image[top:bottom, left:right].copy()
    crop_height, crop_width = crop.shape[:2]
    background_bgr = _estimate_background_bgr(crop)

    crop_float = crop.astype(np.float32)
    background_delta = np.linalg.norm(crop_float - background_bgr.reshape(1, 1, 3), axis=2)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = cv2.magnitude(grad_x, grad_y)

    border_band_y = max(1, int(round(crop_height * 0.08)))
    border_band_x = max(1, int(round(crop_width * 0.08)))
    border_delta = np.concatenate(
        [
            background_delta[:border_band_y, :].reshape(-1),
            background_delta[-border_band_y:, :].reshape(-1),
            background_delta[:, :border_band_x].reshape(-1),
            background_delta[:, -border_band_x:].reshape(-1),
        ]
    )
    border_grad = np.concatenate(
        [
            gradient_mag[:border_band_y, :].reshape(-1),
            gradient_mag[-border_band_y:, :].reshape(-1),
            gradient_mag[:, :border_band_x].reshape(-1),
            gradient_mag[:, -border_band_x:].reshape(-1),
        ]
    )

    delta_threshold = max(18.0, float(np.percentile(border_delta, 95)) * 2.5)
    gradient_threshold = max(14.0, float(np.percentile(border_grad, 95)) * 2.2)

    active_mask = (background_delta >= delta_threshold) | (gradient_mag >= gradient_threshold)
    active_mask = active_mask.astype(np.uint8) * 255
    active_mask = cv2.morphologyEx(active_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    active_mask = cv2.morphologyEx(active_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    active_bool = active_mask > 0

    row_fill_ratio = active_bool.mean(axis=1) if crop_height else np.array([], dtype=np.float32)
    col_fill_ratio = active_bool.mean(axis=0) if crop_width else np.array([], dtype=np.float32)
    smoothed_row_fill = _smooth_profile(row_fill_ratio, max(3, int(round(crop_height * 0.03))))
    smoothed_col_fill = _smooth_profile(col_fill_ratio, max(3, int(round(crop_width * 0.03))))

    row_activation_threshold = max(0.012, min(0.04, 10.0 / max(crop_width, 1)))
    col_activation_threshold = max(0.01, min(0.05, 10.0 / max(crop_height, 1)))
    active_rows = smoothed_row_fill >= row_activation_threshold
    active_cols = smoothed_col_fill >= col_activation_threshold

    row_indices = np.flatnonzero(active_rows)
    col_indices = np.flatnonzero(active_cols)

    if row_indices.size == 0:
        active_top = 0
        active_bottom = 0
        active_height_ratio = 0.0
        active_bottom_ratio = 0.0
    else:
        active_top = int(row_indices[0])
        active_bottom = int(row_indices[-1]) + 1
        active_height_ratio = float(active_bottom - active_top) / float(crop_height)
        active_bottom_ratio = float(active_bottom) / float(crop_height)

    if col_indices.size == 0:
        active_left = 0
        active_right = 0
        active_width_ratio = 0.0
    else:
        active_left = int(col_indices[0])
        active_right = int(col_indices[-1]) + 1
        active_width_ratio = float(active_right - active_left) / float(crop_width)

    lower_band_start = int(round(crop_height * lower_band_start_ratio))
    lower_band_start = min(max(lower_band_start, 0), crop_height)
    lower_band_bool = active_bool[lower_band_start:, :] if lower_band_start < crop_height else active_bool[0:0, :]
    lower_band_rows = active_rows[lower_band_start:] if lower_band_start < crop_height else active_rows[0:0]
    lower_band_active_area_ratio = float(lower_band_bool.mean()) if lower_band_bool.size else 0.0
    lower_band_active_row_ratio = float(lower_band_rows.mean()) if lower_band_rows.size else 0.0
    lower_band_largest_empty_run_ratio = (
        float(_largest_inactive_run(lower_band_rows)) / float(max(lower_band_rows.size, 1))
        if lower_band_rows.size
        else 0.0
    )
    lower_band_grid_occupancy_ratio = _coarse_grid_occupancy_ratio(
        lower_band_bool,
        target_rows=8,
        target_cols=14,
        min_cell_fill_ratio=0.035,
    )

    largest_empty_run_ratio = float(_largest_inactive_run(active_rows)) / float(crop_height)
    active_area_ratio = float(active_bool.mean()) if active_bool.size else 0.0

    return {
        "crop_box": [left, top, right - left, bottom - top],
        "background_bgr": [round(float(value), 2) for value in background_bgr.tolist()],
        "pixel_delta_threshold": round(delta_threshold, 2),
        "gradient_threshold": round(gradient_threshold, 2),
        "row_activation_threshold": round(row_activation_threshold, 4),
        "col_activation_threshold": round(col_activation_threshold, 4),
        "active_area_ratio": round(active_area_ratio, 4),
        "active_height_ratio": round(active_height_ratio, 4),
        "active_width_ratio": round(active_width_ratio, 4),
        "active_bottom_ratio": round(active_bottom_ratio, 4),
        "active_bbox_local": [active_left, active_top, max(active_right - active_left, 0), max(active_bottom - active_top, 0)],
        "lower_band_start_ratio": round(lower_band_start_ratio, 4),
        "lower_band_active_area_ratio": round(lower_band_active_area_ratio, 4),
        "lower_band_active_row_ratio": round(lower_band_active_row_ratio, 4),
        "lower_band_grid_occupancy_ratio": round(lower_band_grid_occupancy_ratio, 4),
        "lower_band_largest_empty_run_ratio": round(lower_band_largest_empty_run_ratio, 4),
        "largest_empty_run_ratio": round(largest_empty_run_ratio, 4),
    }