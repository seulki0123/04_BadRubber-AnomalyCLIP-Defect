import os

import cv2
import numpy as np

from AnomalyCLIP import utils
from AnomalyCLIP.visualization import apply_ad_scoremap

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    # scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def draw_anomaly_map(image, anomaly_map):
    mask = utils.normalize(anomaly_map[0])
    vis = apply_ad_scoremap(image, mask)
    # vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
    return vis

def draw_segmentation(
    image,
    mask,
    color=(255, 0, 0),
    alpha=0.4
):
    """
    image: RGB image (H, W, 3)
    mask: binary mask (H, W)
    """
    overlay = image.copy()

    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color

    overlay = cv2.addWeighted(
        overlay, 1 - alpha,
        colored_mask, alpha,
        0
    )

    return overlay

def draw_segmentation_outline(
    image,
    mask,
    color=(0, 0, 0),
    thickness=1
):
    """
    image: RGB image (H, W, 3)
    mask: binary mask (H, W), uint8 {0,255}
    return: image with polygon outlines
    """
    vis = image.copy()

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        vis,
        contours,
        contourIdx=-1,   # draw all
        color=color,
        thickness=thickness
    )

    return vis

def draw_yolo_segmentation(
    image,
    yolo_masks,
    yolo_boxes=None,
    alpha=0.4
):
    """
    image: (H, W, 3) BGR
    yolo_masks: (N, H, W) binary or {0,1}
    yolo_boxes: (N, 4) optional
    """
    vis = image.copy()

    for i, mask in enumerate(yolo_masks):
        color = np.random.randint(0, 255, size=3).tolist()

        colored_mask = np.zeros_like(vis, dtype=np.uint8)
        colored_mask[mask > 0] = color

        vis = cv2.addWeighted(vis, 1 - alpha, colored_mask, alpha, 0)

        if yolo_boxes is not None:
            x1, y1, x2, y2 = yolo_boxes[i]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    return vis

def draw_bboxes_with_labels(
    image,
    bboxes,
    labels,
    color=(0, 255, 0),
    thickness=1,
    font_scale=0.3
):
    """
    image: RGB image
    bboxes: [(x1,y1,x2,y2)]
    labels: list[str] or list[int]
    """
    vis = image.copy()

    for (x1, y1, x2, y2), label in zip(bboxes, labels):
        # bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        text = str(label)
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        # label background
        cv2.rectangle(
            vis,
            (x1, y1 - th - 6),
            (x1 + tw + 4, y1),
            color,
            -1
        )

        # label text
        cv2.putText(
            vis,
            text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return vis
