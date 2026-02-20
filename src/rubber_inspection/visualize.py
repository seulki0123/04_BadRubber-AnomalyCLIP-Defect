from typing import Sequence, Optional
import numpy as np
import cv2

from outputs import ForegroundMaskBatchItem, AnomalyCLIPBatchItem, ClassificationBatchItem, SegmentationBatchItem


def visualize(
    image: np.ndarray,
    foreground: ForegroundMaskBatchItem,
    anomaly: AnomalyCLIPBatchItem,
    classification: ClassificationBatchItem,
    segmentation: SegmentationBatchItem,
    show_foreground: bool = True,
    show_anomaly_map: bool = True,
    show_anomaly_score: bool = True,
) -> np.ndarray:

    vis_img = image.copy()

    if show_foreground:
        vis_img = draw_normalized_polygons(
            image=vis_img,
            polygons_n=foreground.polygons_n,
            colors=[(0, 255, 0) for _ in range(len(foreground.polygons_n))],
            thickness=1,
        )

    if show_anomaly_map:
        vis_img = draw_anomaly_map(
            image=vis_img,
            anomaly_map=anomaly.map,
            alpha=0.3,
        )
    
    if show_anomaly_score:
        vis_img = put_text_box(
            image=vis_img,
            text=f"Global anomaly: {anomaly.global_score:.3f}",
            position="top_left",
            bg_color=None,
            text_color=(255, 255, 255),
        )

    vis_img = draw_normalized_polygons(
        image=vis_img,
        polygons_n=[region.polygon_n for region in anomaly.regions],
        colors=[region.color for region in classification.regions],
        thickness=1,
    )

    vis_img = draw_bboxes_xyxyn(
        image=vis_img,
        bboxes_xyxyn=[region.bboxes_xyxy_n for region in anomaly.regions],
        labels=[f"{region.class_name} {region.confidence:.2f}" for region in classification.regions],
        colors=[region.color for region in classification.regions],
        thickness=1,
    )
    
    vis_img = draw_normalized_polygons(
        image=vis_img,
        polygons_n=[region.polygon_n for region in segmentation.regions],
        labels=[f"{region.class_name} {region.confidence:.2f}" for region in segmentation.regions],
        colors=[region.color for region in segmentation.regions],
        thickness=1,
    )
    
    return vis_img

def draw_normalized_polygons(
    image: np.ndarray,
    polygons_n: Sequence[np.ndarray],
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[tuple]] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    closed: bool = True,
    alpha: float = 0.2,
) -> np.ndarray:

    vis_img = image.copy()
    overlay = vis_img.copy()
    H, W = vis_img.shape[:2]

    if colors is None:
        colors = [(0, 255, 0)] * len(polygons_n)

    for i, poly_n in enumerate(polygons_n):
        if poly_n.size == 0:
            continue

        color = colors[i] if i < len(colors) else (0, 255, 0)

        poly_px = poly_n.astype(np.float32).copy()
        poly_px[:, 0] *= W
        poly_px[:, 1] *= H
        poly_px = poly_px.astype(np.int32)

        cv2.polylines(
            overlay,
            [poly_px],
            isClosed=closed,
            color=color,
            thickness=thickness,
        )

    vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)

    if labels is not None:
        for i, poly_n in enumerate(polygons_n):
            if poly_n.size == 0 or i >= len(labels):
                continue
            poly_px = poly_n.copy()
            poly_px[:, 0] *= W
            poly_px[:, 1] *= H
            poly_px = poly_px.astype(np.int32)

            draw_label(
                vis_img,
                str(labels[i]),
                tuple(poly_px[0]),
                colors[i],
                font_scale,
                font_thickness,
            )

    return vis_img

def draw_anomaly_map(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.2,
) -> np.ndarray:

    amap = np.clip(anomaly_map, 0.0, 1.0)

    heatmap = (amap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    mask = amap > 0  # (H, W) bool

    image[mask] = (
        image[mask] * (1 - alpha)
        + heatmap[mask] * alpha
    ).astype(np.uint8)

    return image

def draw_bboxes_xyxyn(
    image: np.ndarray,
    bboxes_xyxyn: Sequence[Sequence[float]],
    labels: Optional[Sequence[str]] = None,
    colors: Sequence[tuple] = (0, 0, 255),
    thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    alpha: float = 0.2,
) -> np.ndarray:

    vis_img = image.copy()
    overlay = vis_img.copy()
    H, W = vis_img.shape[:2]

    for i, box in enumerate(bboxes_xyxyn):
        x1n, y1n, x2n, y2n = box

        x1 = int(x1n * W)
        y1 = int(y1n * H)
        x2 = int(x2n * W)
        y2 = int(y2n * H)

        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))

        color = colors[i] if i < len(colors) else (0, 0, 255)

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

    vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)

    if labels:
        for i, box in enumerate(bboxes_xyxyn):
            if i >= len(labels):
                continue
            x1 = int(box[0] * W)
            y1 = int(box[1] * H)
            
            draw_label(
                image=vis_img,
                text=str(labels[i]),
                origin=(x1, y1),
                color=color,
                font_scale=font_scale,
                font_thickness=font_thickness,
            )

    return vis_img

def put_text_box(
    image: np.ndarray,
    text: str,
    position: str = "top_left",
    margin: int = 10,
    font_scale: float = 0.6,
    font_thickness: int = 1,
    text_color: tuple = (255, 255, 255),
    bg_color: Optional[tuple] = (0, 0, 0),
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Draw text with background box at a predefined position.

    position:
        - top_left
        - top_right
        - bottom_left
        - bottom_right
        - center
    """

    vis = image.copy()
    H, W = vis.shape[:2]

    (tw, th), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness,
    )

    box_w = tw + 8
    box_h = th + baseline + 8

    if position == "top_left":
        x1 = margin
        y1 = margin

    elif position == "top_right":
        x1 = W - box_w - margin
        y1 = margin

    elif position == "bottom_left":
        x1 = margin
        y1 = H - box_h - margin

    elif position == "bottom_right":
        x1 = W - box_w - margin
        y1 = H - box_h - margin

    elif position == "center":
        x1 = (W - box_w) // 2
        y1 = (H - box_h) // 2

    else:
        raise ValueError(f"Unknown position: {position}")

    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = vis.copy()

    if bg_color is not None:
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            bg_color,
            -1,
        )

    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

    text_x = x1 + 4
    text_y = y1 + box_h - baseline - 4

    cv2.putText(
        vis,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    return vis

def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple,              # (x, y) pixel
    color: tuple,
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> None:
    """
    Draw text label only (no background box).
    Text color = same as region color.
    Modifies image in-place.
    """

    H, W = image.shape[:2]
    x, y = origin

    (tw, th), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness,
    )

    text_y = y - 4

    if text_y - th < 0:
        text_y = y + th + 4

    text_x = max(0, min(W - tw, x))
    text_y = max(th, min(H - baseline, text_y))

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        font_thickness,
        cv2.LINE_AA,
    )
