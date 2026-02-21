import hashlib
import random
from typing import Tuple, Optional

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

def compute_image_hash(img: np.ndarray) -> str:
    return hashlib.md5(img.tobytes()).hexdigest()

def save_polygons_to_yolo_format(
    save_path: str,
    polygons: list,
    class_ids: list,
) -> None:

    if not polygons:
        return

    lines = []

    for polygon_n, class_id in zip(polygons, class_ids):

        if polygon_n is None:
            continue

        polygon_n = np.asarray(polygon_n, dtype=np.float32)

        if polygon_n.ndim != 2 or polygon_n.shape[0] < 3:
            continue

        coords = polygon_n.reshape(-1)
        coord_str = " ".join(f"{c:.6f}" for c in coords)
        lines.append(f"{class_id} {coord_str}")

    if not lines:
        return

    with open(save_path, "w") as f:
        f.write("\n".join(lines))