import random
from typing import Tuple

import cv2
import numpy as np


def scale_bbox_xyxy_n(
    bbox: Tuple[float, float, float, float],
    scale: float = 2.0,
) -> Tuple[float, float, float, float]:

    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale

    x1_s = np.clip(cx - w / 2, 0.0, 1.0)
    y1_s = np.clip(cy - h / 2, 0.0, 1.0)
    x2_s = np.clip(cx + w / 2, 0.0, 1.0)
    y2_s = np.clip(cy + h / 2, 0.0, 1.0)

    return x1_s, y1_s, x2_s, y2_s

def mask_to_polygon(mask: np.ndarray) -> np.ndarray:
    """
    mask: (H, W) uint8 or bool
    return: (N, 2) polygon
    """

    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,      # 가장 바깥 contour만
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return np.zeros((0, 2))

    # 가장 큰 contour 선택
    contour = max(contours, key=cv2.contourArea)

    # (N,1,2) → (N,2)
    polygon = contour.squeeze(1)

    return polygon

def normalize_polygon(polygon: np.ndarray, height: int, width: int):
    polygon = polygon.astype(np.float32)
    polygon[:, 0] /= width   # x / W
    polygon[:, 1] /= height  # y / H
    return polygon

def random_color(
    bright: bool = True,
) -> Tuple[int, int, int]:
    """
    Return random BGR color.
    bright=True → 밝은 색 위주
    """

    if bright:
        return (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255),
        )
    else:
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )