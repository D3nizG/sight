from __future__ import annotations

from typing import Callable, Dict

import cv2
import numpy as np


# Ordered processing modes for cycling in the UI
PROCESSING_MODES = [
    "none",
    "gray",
    "blur",
    "edges",
    "binary",
    "sobel",
]


def to_bgr(single_channel: np.ndarray) -> np.ndarray:
    """Convert a single-channel image to BGR for consistent rendering."""
    return cv2.cvtColor(single_channel, cv2.COLOR_GRAY2BGR)


def process_none(frame_bgr: np.ndarray) -> np.ndarray:
    return frame_bgr


def process_gray(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return to_bgr(gray)


def process_blur(frame_bgr: np.ndarray) -> np.ndarray:
    # Gaussian blur with a moderate kernel size
    return cv2.GaussianBlur(frame_bgr, (9, 9), 0)


def process_edges(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Use median as robust threshold baseline
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lower, upper)
    edges_colored = to_bgr(edges)
    # Overlay edges on a dimmed original for context
    dimmed = (frame_bgr * 0.4).astype(np.uint8)
    return cv2.addWeighted(edges_colored, 1.0, dimmed, 1.0, 0)


def process_binary(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold is robust to lighting changes
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
    )
    return to_bgr(binary)


def process_sobel(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return to_bgr(sobel)


MODE_TO_FUNCTION: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": process_none,
    "gray": process_gray,
    "blur": process_blur,
    "edges": process_edges,
    "binary": process_binary,
    "sobel": process_sobel,
}


def apply_processing(frame_bgr: np.ndarray, mode: str) -> np.ndarray:
    """Apply a processing mode to a BGR frame.

    Always returns a BGR image of the same size as input.
    """
    func = MODE_TO_FUNCTION.get(mode, process_none)
    try:
        return func(frame_bgr)
    except Exception:
        # If processing fails for any reason, fall back to original frame
        return frame_bgr


