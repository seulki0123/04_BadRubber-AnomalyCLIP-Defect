import numpy as np

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
