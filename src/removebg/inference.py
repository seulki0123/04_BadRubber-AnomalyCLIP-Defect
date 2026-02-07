from typing import Sequence

import cv2
import tqdm
import numpy as np
from ultralytics import YOLO


class BackgroundRemover:
    def __init__(self, checkpoint_path: str, imgsz: int) -> None:
        self.model = YOLO(checkpoint_path)
        self.imgsz = imgsz
        self._warmup()

    def _warmup(
        self,
        batch_size: int = 1,
    ) -> None:
        for _ in tqdm.tqdm(range(10), desc="Warm up YOLO segmentation model"):
            dummy_images = [
                np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                for _ in range(batch_size)
            ]
            _ = self.model(dummy_images, imgsz=self.imgsz, verbose=False)

    def remove(
        self,
        images: Sequence[np.ndarray],
    ) -> np.ndarray:
        results = self.model(images, imgsz=self.imgsz, verbose=False)
        return self.yolo_results_to_masks(results)

    @staticmethod
    def yolo_results_to_masks(results) -> np.ndarray:
        masks = []

        for r in results:
            H, W = r.orig_shape

            if r.masks is None:
                masks.append(np.ones((H, W), dtype=np.float32))
                continue

            fg = r.masks.data.any(dim=0).float().cpu().numpy()

            fg_resized = cv2.resize(
                fg,
                (W, H),
                interpolation=cv2.INTER_NEAREST,
            )

            masks.append(fg_resized.astype(np.float32))

        return np.stack(masks)  # (B, H_orig, W_orig)
