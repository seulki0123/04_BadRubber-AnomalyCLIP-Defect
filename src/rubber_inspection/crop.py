from typing import Dict, Tuple
import numpy as np

from outputs import (
    AnomalyCLIPBatchItem,
    ClassificationBatchItem,
    SegmentationBatchItem,
)
from utils import scale_bbox_xyxy_n
from .visualize import draw_normalized_polygons

def crop_regions(
    image: np.ndarray,
    regions,
    classes,
    prefix: str,
    name_from: str = "cls",
    draw_polygon: bool = False,
    polygon_thickness: int = 1,
    polygon_alpha: float = 0.3,
) -> Dict[str, np.ndarray]:

    if len(regions) != len(classes):
        raise ValueError("regions and classes length mismatch")

    H, W = image.shape[:2]
    crops: Dict[str, np.ndarray] = {}

    for idx, (region, cls) in enumerate(zip(regions, classes)):

        x1n, y1n, x2n, y2n = scale_bbox_xyxy_n(region.bboxes_xyxy_n, scale=2.0)

        x1 = max(0, min(W, int(x1n * W)))
        y1 = max(0, min(H, int(y1n * H)))
        x2 = max(0, min(W, int(x2n * W)))
        y2 = max(0, min(H, int(y2n * H)))

        if x2 <= x1 or y2 <= y1:
            continue

        crop_img = image[y1:y2, x1:x2].copy()

        if draw_polygon and hasattr(region, "polygon_n") and region.polygon_n.size > 0:

            poly_n = region.polygon_n.copy()

            poly_px = poly_n.copy()
            poly_px[:, 0] *= W
            poly_px[:, 1] *= H

            poly_px[:, 0] -= x1
            poly_px[:, 1] -= y1

            crop_h, crop_w = crop_img.shape[:2]
            poly_px[:, 0] /= crop_w
            poly_px[:, 1] /= crop_h

            crop_img = draw_normalized_polygons(
                image=crop_img,
                polygons_n=[poly_px],
                colors=[cls.color if hasattr(cls, "color") else (0, 255, 0)],
                thickness=polygon_thickness,
                alpha=polygon_alpha,
            )

        if name_from == "region" and hasattr(region, "class_name"):
            class_name = region.class_name
        else:
            class_name = cls.class_name

        filename = f"{prefix}_{class_name}_{cls.confidence:.2f}_{idx}.jpg"
        crops[filename] = crop_img

    return crops