import os

import cv2
import numpy as np

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


def black_pixel_ratio(
    image,
    threshold=10
):
    """
    image: np.ndarray (H, W, 3), BGR or RGB
    return: ratio (0~1)
    """
    # 모든 채널이 threshold 이하인 픽셀
    black_mask = np.all(image <= threshold, axis=2)

    black_pixels = black_mask.sum()
    total_pixels = image.shape[0] * image.shape[1]

    return black_pixels / total_pixels

def anomaly_overlap_ratio(anomaly_mask, yolo_mask):
    """
    anomaly_mask: (H, W) uint8 {0,255}
    yolo_mask:    (H, W) bool or {0,1}
    """
    anomaly_bin = anomaly_mask > 0
    yolo_bin = yolo_mask > 0

    anomaly_area = anomaly_bin.sum()
    if anomaly_area == 0:
        return 0.0

    intersection = (anomaly_bin & yolo_bin).sum()
    return intersection / anomaly_area

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
