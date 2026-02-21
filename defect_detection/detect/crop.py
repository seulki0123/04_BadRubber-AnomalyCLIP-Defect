import os
from typing import Dict, Tuple, Optional, Any, List
import numpy as np

from defect_detection.utils import scale_bbox_xyxy_n, compute_image_hash
from .visualize import draw_normalized_polygons


# ============================================================
# Basic Geometry Utilities
# ============================================================

def denormalize_bbox_xyxy_n(bbox_n, H, W):
    x1n, y1n, x2n, y2n = bbox_n
    return [
        int(x1n * W),
        int(y1n * H),
        int(x2n * W),
        int(y2n * H),
    ]


def compute_crop_bbox(region, H: int, W: int, scale: float = 2.0) -> Optional[Tuple[int, int, int, int]]:
    x1n, y1n, x2n, y2n = scale_bbox_xyxy_n(region.bboxes_xyxy_n, scale=scale)

    x1 = max(0, min(W, int(x1n * W)))
    y1 = max(0, min(H, int(y1n * H)))
    x2 = max(0, min(W, int(x2n * W)))
    y2 = max(0, min(H, int(y2n * H)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2].copy()


# ============================================================
# Polygon Conversion
# ============================================================

def convert_polygon_to_crop_normalized(
    polygon_n: np.ndarray,
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> Optional[np.ndarray]:

    if polygon_n is None or polygon_n.size == 0:
        return None

    H, W = image_shape
    x1, y1, x2, y2 = bbox
    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w <= 0 or crop_h <= 0:
        return None

    poly_px = polygon_n.copy()
    poly_px[:, 0] *= W
    poly_px[:, 1] *= H

    poly_px[:, 0] -= x1
    poly_px[:, 1] -= y1

    poly_px[:, 0] /= crop_w
    poly_px[:, 1] /= crop_h

    poly_px = poly_px.clip(0, 1)

    if len(poly_px) < 3:
        return None

    return poly_px.astype(np.float32)


# ============================================================
# Segmentation Processing
# ============================================================

def process_segmentations(
    seg_list,
    bbox_scaled,
    image_shape,
):
    polygons_to_draw = []
    segmentations = []

    total_area = 0
    max_conf = 0.0
    dominant_area = 0
    dominant_class_name = None

    for seg in seg_list:
        polygon_n = convert_polygon_to_crop_normalized(
            seg.polygon_n,
            bbox_scaled,
            image_shape,
        )

        if polygon_n is None:
            continue

        polygons_to_draw.append((polygon_n, seg.color))

        seg_area = int(seg.area)
        seg_conf = float(seg.confidence)

        total_area += seg_area

        if seg_conf > max_conf:
            max_conf = seg_conf

        if seg_area > dominant_area:
            dominant_area = seg_area
            dominant_class_name = seg.class_name

        segmentations.append({
            "class_id": seg.class_id,
            "class_name": seg.class_name,
            "confidence": seg_conf,
            "area": seg_area,
            "polygon": polygon_n.tolist(),
            "color": seg.color,
        })

    return (
        polygons_to_draw,
        segmentations,
        total_area,
        max_conf,
        dominant_class_name,
    )


# ============================================================
# Filename
# ============================================================

def build_filename(*parts) -> str:
    cleaned = []
    for p in parts:
        if p is None:
            continue
        s = str(p).strip()
        if s:
            cleaned.append(s)
    return "_".join(cleaned)


# ============================================================
# Main Pipeline
# ============================================================

def crop_regions(
    image: np.ndarray,
    imagename: str,
    crop_sources,
    polygon_sources,
    prefix: str = None,
    draw_polygon: bool = False,
    polygon_thickness: int = 1,
    polygon_alpha: float = 0.3,
) -> Tuple[Dict[str, Tuple[np.ndarray, list]], Dict[str, Any]]:

    H, W = image.shape[:2]

    crops = {}

    metadata = {
        "image": {
            "file": imagename + ".jpg",
            "hash": compute_image_hash(image),
            "resolution": [W, H],
        },
        "crop": []
    }

    for r_idx, (crop_region, seg_list) in enumerate(zip(crop_sources, polygon_sources)):

        bbox_scaled = compute_crop_bbox(crop_region, H, W)
        if bbox_scaled is None:
            continue

        crop_img = crop_image(image, bbox_scaled)

        bbox_original = denormalize_bbox_xyxy_n(
            crop_region.bboxes_xyxy_n,
            H,
            W,
        )

        (
            polygons_to_draw,
            segmentations,
            total_area,
            max_conf,
            dominant_class_name,
        ) = process_segmentations(
            seg_list,
            bbox_scaled,
            (H, W),
        )

        if not segmentations:
            continue

        if draw_polygon:
            crop_img = draw_normalized_polygons(
                image=crop_img,
                polygons_n=[p[0] for p in polygons_to_draw],
                colors=[p[1] for p in polygons_to_draw],
                thickness=polygon_thickness,
                alpha=polygon_alpha,
            )

        filename = build_filename(
            prefix,
            dominant_class_name if dominant_class_name else "none",
            f"{int(total_area):010d}",
            f"{max_conf:.4f}",
            f"{r_idx:02d}",
            imagename,
        )

        crop_hash = compute_image_hash(crop_img)

        x1, y1, x2, y2 = bbox_scaled
        crop_w = x2 - x1
        crop_h = y2 - y1

        metadata["crop"].append({
            "file": filename + ".jpg",
            "hash": crop_hash,
            "xyxy": list(bbox_scaled),
            "xyxy_original": [int(v) for v in bbox_original],
            "resolution": [crop_w, crop_h],
            "area": crop_w * crop_h,
            "segmentations": segmentations,
        })

        crops[filename] = (crop_img, segmentations)

    return crops, metadata