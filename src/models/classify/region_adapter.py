from typing import List, Tuple
import numpy as np

from outputs import RegionClassificationOutput, Classification
from outputs.anomalyclip import AnomalyCLIPOutput
from utils import scale_bbox_xyxy_n
from .inference import Classifier


class RegionClassifierAdapter:
    """
    AnomalyCLIPOutput → RegionClassificationOutput
    """

    def __init__(self, classifier: Classifier):
        self.classifier = classifier

    def infer(
        self,
        images: List[np.ndarray],
        anomaly: AnomalyCLIPOutput,
    ) -> RegionClassificationOutput:

        patches = []
        mapping: List[Tuple[int, int]] = []

        for b_idx, (img, regions) in enumerate(zip(images, anomaly.batch_regions)):
            H, W = img.shape[:2]

            for r_idx, region in enumerate(regions):
                x1n, y1n, x2n, y2n = scale_bbox_xyxy_n(region.bboxes_xyxy_n, scale=2.0)

                x1, y1 = int(x1n * W), int(y1n * H)
                x2, y2 = int(x2n * W), int(y2n * H)

                if x2 <= x1 or y2 <= y1:
                    continue

                patch = img[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                patches.append(patch)
                mapping.append((b_idx, r_idx))

        results = self.classifier.infer_patches(patches)

        batch_out = [
            [None] * len(regions)
            for regions in anomaly.batch_regions
        ]

        for (b_idx, r_idx), cls in zip(mapping, results):
            batch_out[b_idx][r_idx] = cls

        for b in range(len(batch_out)):
            for r in range(len(batch_out[b])):
                if batch_out[b][r] is None:
                    batch_out[b][r] = Classification("unknown", 0.0)

        return RegionClassificationOutput(batch_out)
