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

def get_segmentation_mask(
    anomaly_map,
    threshold=None,
    normalize=True
):
    """
    anomaly_map: (H, W) or (1, H, W)
    threshold: float or None (자동 threshold)
    return: binary mask (H, W), uint8 {0,255}
    """
    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map[0]

    amap = anomaly_map.astype(np.float32)

    if normalize:
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

    if threshold is None:
        # Otsu threshold
        _, mask = cv2.threshold(
            (amap * 255).astype(np.uint8),
            0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        mask = (amap > threshold).astype(np.uint8) * 255

    return mask

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


def get_bboxes_from_mask(
    mask,
    min_area=100
):
    """
    mask: binary mask (H, W)
    return: list of bboxes [(x1,y1,x2,y2)]
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, x + w, y + h))

    return bboxes

def draw_bboxes(
    image,
    bboxes,
    color=(0, 255, 0),
    thickness=1
):
    """
    image: RGB image
    bboxes: [(x1,y1,x2,y2)]
    """
    vis = image.copy()

    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

    return vis

def crop_bboxes(
    image,
    bboxes,
    scale=1.2,
    min_size=None,
    save_dir=None,
    prefix="crop",
    clamp=True
):
    """
    image: np.ndarray (H, W, 3)
    bboxes: [(x1,y1,x2,y2)]
    scale: float (bbox 크기 대비 확장 비율, 1.0 = 그대로)
    min_size: int or None (crop 최소 크기)
    save_dir: str or None (저장 안 하면 list로 반환)
    prefix: 저장 파일 prefix
    clamp: image boundary 넘어가면 clamp 여부

    return:
        crops: list of np.ndarray (save_dir=None일 때)
    """

    H, W = image.shape[:2]
    crops = []

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for idx, (x1, y1, x2, y2) in enumerate(bboxes):
        bw = x2 - x1
        bh = y2 - y1

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        new_w = bw * scale
        new_h = bh * scale

        if min_size is not None:
            new_w = max(new_w, min_size)
            new_h = max(new_h, min_size)

        nx1 = int(cx - new_w / 2)
        ny1 = int(cy - new_h / 2)
        nx2 = int(cx + new_w / 2)
        ny2 = int(cy + new_h / 2)

        if clamp:
            nx1 = max(0, nx1)
            ny1 = max(0, ny1)
            nx2 = min(W, nx2)
            ny2 = min(H, ny2)

        crop = image[ny1:ny2, nx1:nx2]

        if crop.size == 0:
            continue

        if save_dir is not None:
            save_path = os.path.join(
                save_dir, f"{prefix}_{idx}.jpg"
            )
            cv2.imwrite(save_path, crop)
        else:
            crops.append(crop)

    return crops

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
