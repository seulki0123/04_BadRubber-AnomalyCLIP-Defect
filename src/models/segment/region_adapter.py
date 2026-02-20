from typing import List, Tuple
import numpy as np

from outputs import SegmentationOutput, Segmentation
from outputs.anomalyclip import AnomalyCLIPOutput
from outputs.classify import RegionClassificationOutput
from utils import scale_bbox_xyxy_n
from .inference import Segmenter


class RegionSegmenterAdapter:
    """
    AnomalyCLIPOutput → SegmentationOutput (flattened per image)
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
        mapping: List[int] = []
        offsets: List[Tuple[int, int, int, int]] = []

        for b_idx, (img, regions) in enumerate(zip(images, anomaly.batch_regions)):
            H, W = img.shape[:2]

            for region, region_cls in zip(regions, classifications[b_idx].regions):
                # if region_cls.is_pass:
                #     continue
                
                x1n, y1n, x2n, y2n = scale_bbox_xyxy_n(region.bboxes_xyxy_n, scale=2.0)
                

                x1, y1 = int(x1n * W), int(y1n * H)
                x2, y2 = int(x2n * W), int(y2n * H)

                if x2 <= x1 or y2 <= y1:
                    continue

                patch = img[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                patches.append(patch)
                mapping.append(b_idx)
                offsets.append((x1, y1, W, H))

        results = self.segmenter.infer_patches(patches, offsets)

        batch_out: List[List[Segmentation]] = [
            [] for _ in images
        ]

        for img_idx, segs in zip(mapping, results):
            batch_out[img_idx].extend(segs)

        return SegmentationOutput(batch_out)
