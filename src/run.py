import os

import cv2

from inspector import AnomalyInspector
from classify import visualize

if __name__ == "__main__":
    anomaly_inspector = AnomalyInspector()

    img_dir = "./tests/dot"
    imgs_path = [os.path.join(img_dir, img_path) for img_path in os.listdir(img_dir) if img_path.endswith(".jpg")]
    dst_dir = "./tests/dot_results"
    os.makedirs(dst_dir, exist_ok=True)

    # imgs_path = (
    #     "./tests/SSBR_1_20251227_000101_701.jpg",
    #     "./tests/SSBR_1_20260204_000027_617.jpg"
    # )

    # 1. inspect
    results = anomaly_inspector.inspect(imgs_path)

    for i, res in enumerate(results):
        vis_img, crops_img = visualize(
            result=res,
            draw_anomaly_map=False
        )

        save_name = os.path.basename(imgs_path[i])
        cv2.imwrite(os.path.join(dst_dir, save_name), vis_img)