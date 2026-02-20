from dataclasses import dataclass
from typing import Sequence, Iterator
import numpy as np

from outputs import (
    ForegroundMaskOutput,
    AnomalyCLIPOutput,
    RegionClassificationOutput,
    ClassificationBatchItem,
    RegionSegmentationOutput,
    SegmentationBatchItem,
    AnomalyCLIPBatchItem,
    ForegroundMaskBatchItem,
)
from utils import load_config
from .visualize import visualize
vis_show_cfg = load_config()["show"]

# ---------------------------------
# Batch Item
# ---------------------------------

@dataclass
class InspectorBatchItem:
    image: np.ndarray
    image_path: str
    foreground: ForegroundMaskBatchItem
    anomaly: AnomalyCLIPBatchItem
    classification: ClassificationBatchItem
    segmentation: SegmentationBatchItem

    @property
    def regions(self):
        return self.anomaly.regions

    def visualize(self) -> np.ndarray:
        return visualize(
            image=self.image,
            foreground=self.foreground,
            anomaly=self.anomaly,
            classification=self.classification,
            segmentation=self.segmentation,
            show_foreground=vis_show_cfg["foreground"],
            show_anomaly_map=vis_show_cfg["anomaly_map"],
            show_anomaly_score=vis_show_cfg["anomaly_score"],
        )


# ---------------------------------
# Output
# ---------------------------------

@dataclass
class InspectorOutput:
    images: Sequence[np.ndarray]
    images_path: Sequence[str]
    foreground: ForegroundMaskOutput
    anomaly: AnomalyCLIPOutput
    classification: RegionClassificationOutput
    segmentation: RegionSegmentationOutput
    
    def __post_init__(self):
        B = len(self.images)

        if len(self.foreground) != B:
            raise ValueError("Foreground batch size mismatch")

        if len(self.anomaly) != B:
            raise ValueError("Anomaly batch size mismatch")

        if len(self.classification) != B:
            raise ValueError("Classification batch size mismatch")

        if len(self.segmentation) != B:
            raise ValueError("Segmentation batch size mismatch")

    def __len__(self):
        return len(self.images)

    def __iter__(self) -> Iterator[InspectorBatchItem]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> InspectorBatchItem:
        return InspectorBatchItem(
            image=self.images[idx],
            image_path=self.images_path[idx],
            foreground=self.foreground[idx],
            anomaly=self.anomaly[idx],
            classification=self.classification[idx],
            segmentation=self.segmentation[idx],
        )