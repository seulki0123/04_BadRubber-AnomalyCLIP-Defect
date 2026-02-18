from typing import Sequence, Optional
import numpy as np
import cv2

from outputs import ForegroundMaskBatchItem, AnomalyCLIPBatchItem, ClassificationBatchItem


def visualize(
    image: np.ndarray,
    foreground: ForegroundMaskBatchItem,
    anomaly: AnomalyCLIPBatchItem,
    classification: ClassificationBatchItem,
    show_foreground: bool = True,
    show_anomaly_map: bool = True,
    show_anomaly_score: bool = True,
) -> np.ndarray:

    vis_img = image.copy()

    if show_foreground:
        vis_img = draw_normalized_polygons(
            image=vis_img,
            polygons_n=foreground.polygons_n,
            color=(0, 255, 0),
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
        color=(0, 0, 255),
        thickness=1,
    )

    vis_img = draw_bboxes_xyxyn(
        image=vis_img,
        bboxes_xyxyn=[region.bboxes_xyxy_n for region in anomaly.regions],
        labels=[f"{region.class_name} {region.confidence:.2f}" for region in classification.regions],
        colors=[region.color for region in classification.regions],
        thickness=1,
    )
    
    return vis_img


def draw_normalized_polygons(
    image: np.ndarray,
    polygons_n: Sequence[np.ndarray],
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    closed: bool = True,
) -> np.ndarray:
    """
    Draw normalized (0~1) polygons on image.

    polygons_n: list of (N,2) normalized polygons
    """
    vis_img = image.copy()
    H, W = vis_img.shape[:2]

    for poly_n in polygons_n:
        if poly_n.size == 0:
            continue

        poly_px = poly_n.copy()
        poly_px[:, 0] *= W
        poly_px[:, 1] *= H
        poly_px = poly_px.astype(np.int32)

        cv2.polylines(
            vis_img,
            [poly_px],
            isClosed=closed,
            color=color,
            thickness=thickness,
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
    bboxes_xyxyn: Sequence[Sequence[float]],  # [(x1,y1,x2,y2), ...] normalized
    labels: Optional[Sequence[str]] = None,
    colors: Sequence[tuple] = (0, 0, 255),
    thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> np.ndarray:
    """
    Draw normalized (x1, y1, x2, y2) bounding boxes with optional labels.
    Coordinates must be in range [0, 1].
    """

    H, W = image.shape[:2]

    for i, box in enumerate(bboxes_xyxyn):
        x1n, y1n, x2n, y2n = box

        # convert to pixel
        x1 = int(x1n * W)
        y1 = int(y1n * H)
        x2 = int(x2n * W)
        y2 = int(y2n * H)

        # clamp
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))

        # rectangle
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            colors[i],
            thickness,
        )

        # label
        if labels is not None:
            label = str(labels[i])

            (tw, th), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness,
            )

            text_y = max(0, y1 - th - baseline - 4)

            # background
            cv2.rectangle(
                image,
                (x1, text_y),
                (x1 + tw + 4, y1),
                colors[i],
                -1,
            )

            cv2.putText(
                image,
                label,
                (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

    return image

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

    # semi-transparent background
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

    # text
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
