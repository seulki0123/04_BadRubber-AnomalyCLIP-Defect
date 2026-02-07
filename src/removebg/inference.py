from typing import Sequence
import numpy as np
from ultralytics import YOLO


class BackgroundRemover:
    def __init__(self, checkpoint_path: str, imgsz: int) -> None:
        self.model = YOLO(checkpoint_path)
        self.imgsz = imgsz

    def remove(
        self,
        image: Sequence[np.ndarray],
        verbose: bool = True,
    ) -> np.ndarray:
        results = self.model(image, imgsz=self.imgsz, verbose=verbose)
        return self.yolo_results_to_masks(results, self.imgsz)

    @staticmethod
    def yolo_results_to_masks(
        results,
        imgsz: int,
    ) -> np.ndarray:
        masks = []

        for r in results:
            if r.masks is None:
                masks.append(np.ones((imgsz, imgsz), dtype=np.float32))
            else:
                fg = r.masks.data.any(dim=0).float()
                masks.append(fg.cpu().numpy())

        return np.stack(masks)  # (B, H, W)
