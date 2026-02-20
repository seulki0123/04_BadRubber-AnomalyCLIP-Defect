from typing import List, Sequence, Tuple

import cv2
import tqdm
import numpy as np
from ultralytics import YOLO

from outputs import Segmentation


class Segmenter:
    def __init__(
        self,
        checkpoint_path: str,
        imgsz: int = 32,
        conf_threshold: float = 0.5,
    ) -> None:
        self.model = YOLO(checkpoint_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self._warmup()

    def _warmup(self, batch_size: int = 1) -> None:
        for _ in tqdm.tqdm(range(5), desc="Warm up YOLO segmenter model for region segmenter"):
            dummy = [np.zeros((self.imgsz, self.imgsz, 3), np.uint8)]
            _ = self.model(dummy, imgsz=self.imgsz, verbose=False)

    def infer_patches(
        self,
        patches: Sequence[np.ndarray],
        offsets: Sequence[Tuple[int, int, int, int]],  # x1, y1, W, H
    ) -> List[List[Segmentation]]:

        if len(patches) == 0:
            return []

        results = self.model(
            patches,
            imgsz=self.imgsz,
            verbose=False,
        )

        outputs: List[List[Segmentation]] = []

        for r, (x1, y1, W, H) in zip(results, offsets):

            region_segments: List[Segmentation] = []

            if r.masks is None:
                outputs.append(region_segments)
                continue

            for mask, cls_id, conf in zip(
                r.masks.xy,
                r.boxes.cls,
                r.boxes.conf,
            ):
                conf = float(conf)
                if conf < self.conf_threshold:
                    continue

                polygon_patch = np.array(mask)

                polygon_global = polygon_patch.copy()
                polygon_global[:, 0] += x1
                polygon_global[:, 1] += y1

                polygon_n = polygon_global.copy()
                polygon_n[:, 0] /= W
                polygon_n[:, 1] /= H

                xmin_n, ymin_n = polygon_n.min(axis=0)
                xmax_n, ymax_n = polygon_n.max(axis=0)

                bboxes_xyxy_n = (
                    max(0.0, min(1.0, float(xmin_n))),
                    max(0.0, min(1.0, float(ymin_n))),
                    max(0.0, min(1.0, float(xmax_n))),
                    max(0.0, min(1.0, float(ymax_n))),
                )
                
                area = cv2.contourArea(polygon_global.astype(np.float32))
                
                cls_id = int(cls_id)

                region_segments.append(
                    Segmentation(
                        polygon_n=polygon_n,
                        bboxes_xyxy_n=bboxes_xyxy_n,
                        confidence=conf,
                        area=area,
                        class_id=cls_id,
                        class_name=self.model.names[cls_id],
                        color=(0, 255, 0),
                    )
                )

            outputs.append(region_segments)

        return outputs
