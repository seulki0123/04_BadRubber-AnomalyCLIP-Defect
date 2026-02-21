from typing import List, Tuple
import numpy as np

from defect_detection.outputs import SegmentationOutput, Segmentation
from defect_detection.outputs.anomalyclip import AnomalyCLIPOutput
from defect_detection.outputs.classify import RegionClassificationOutput
from defect_detection.utils import scale_bbox_xyxy_n
from .inference import Segmenter


class RegionSegmenterAdapter:
    """
    AnomalyCLIPOutput → SegmentationOutput
    [B][R][S] (B: batch size, R: region size, S: segmentation size)
    """

    def __init__(self, segmenter: Segmenter):
        self.segmenter = segmenter

    def infer(
        self,
        images: List[np.ndarray],
        anomaly: AnomalyCLIPOutput,
        classifications: RegionClassificationOutput,
    ) -> SegmentationOutput:

        patches = []
        mapping: List[Tuple[int, int]] = []  # (batch_idx, region_idx)
        offsets: List[Tuple[int, int, int, int]] = []

        # 1. collect patches
        for b_idx, (img, regions) in enumerate(
            zip(images, anomaly.batch_regions)
        ):
            H, W = img.shape[:2]

            for r_idx, (region, region_cls) in enumerate(
                zip(regions, classifications[b_idx].regions)
            ):
                x1n, y1n, x2n, y2n = scale_bbox_xyxy_n(
                    region.bboxes_xyxy_n, scale=2.0
                )

                x1, y1 = int(x1n * W), int(y1n * H)
                x2, y2 = int(x2n * W), int(y2n * H)

                if x2 <= x1 or y2 <= y1:
                    continue

                patch = img[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                patches.append(patch)
                mapping.append((b_idx, r_idx))
                offsets.append((x1, y1, W, H))

        # 2. segmentation inference
        results = self.segmenter.infer_patches(patches, offsets)

        # 3. create [B][R][S] structure
        batch_out: List[List[List[Segmentation]]] = []

        for b_idx in range(len(images)):
            num_regions = len(anomaly.batch_regions[b_idx])
            batch_out.append([[] for _ in range(num_regions)])

        # 4. restore mapping
        for (b_idx, r_idx), segs in zip(mapping, results):
            batch_out[b_idx][r_idx] = segs

        return SegmentationOutput(batch_out)