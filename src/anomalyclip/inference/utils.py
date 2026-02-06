import cv2
import numpy as np
from anomalyclip.utils import normalize


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def visualize(img_np, anomaly_map, save_path):
    vis = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    mask = normalize(anomaly_map[0])
    vis = apply_ad_scoremap(vis, mask)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    print(f"Saved: {save_path}")
    cv2.imwrite(save_path, vis)