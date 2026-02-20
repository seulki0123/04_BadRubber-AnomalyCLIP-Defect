from typing import Sequence, Tuple, List

import cv2
import tqdm
import numpy as np
from ultralytics import YOLO

from outputs.removebg.output import ForegroundMaskOutput


class BackgroundRemover:
    def __init__(self, checkpoint_path: str, imgsz: int) -> None:
        self.model = YOLO(checkpoint_path)
        self.imgsz = imgsz
        self._warmup()

    def _warmup(
        self,
        batch_size: int = 1,
    ) -> None:
        for _ in tqdm.tqdm(range(10), desc="Warm up YOLO segmentation model for background remover"):
            dummy_images = [
                np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                for _ in range(batch_size)
            ]
            _ = self.model(dummy_images, imgsz=self.imgsz, verbose=False)

    def _parse_yolo_segmentation(
        self,
        results,
    ) -> Tuple[np.ndarray, List[List[np.ndarray]]]:

        masks_batch = []
        polygons_batch = []

        for r in results:
            H, W = r.orig_shape

            if r.masks is None:
                masks_batch.append(np.ones((H, W), dtype=np.float32))
                polygons_batch.append([])
                continue

            # --- binary mask (pixel space) ---
            fg = r.masks.data.any(dim=0).float().cpu().numpy()
            fg_resized = cv2.resize(
                fg,
                (W, H),
                interpolation=cv2.INTER_NEAREST,
            )
            masks_batch.append(fg_resized.astype(np.float32))

            # --- normalized polygons (0~1) ---
            polys_n = []
            for poly_n in r.masks.xyn:   # already normalized
                poly_n = np.asarray(poly_n, dtype=np.float32)
                poly_n = np.clip(poly_n, 0.0, 1.0)
                polys_n.append(poly_n)

            polygons_batch.append(polys_n)

        return np.stack(masks_batch), polygons_batch


    def infer(
        self,
        images: Sequence[np.ndarray],
    ) -> ForegroundMaskOutput:
        results = self.model(images, imgsz=self.imgsz, verbose=False)
        masks, polygons_n = self._parse_yolo_segmentation(results)
        return ForegroundMaskOutput(masks=masks, polygons_n=polygons_n)