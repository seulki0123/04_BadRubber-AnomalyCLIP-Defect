from dataclasses import dataclass
from typing import Sequence, Iterator
import numpy as np

from defect_detection.outputs import (
    ForegroundMaskOutput,
    AnomalyCLIPOutput,
    RegionClassificationOutput,
    ClassificationBatchItem,
    SegmentationOutput,
    SegmentationBatchItem,
    AnomalyCLIPBatchItem,
    ForegroundMaskBatchItem,
)
from defect_detection.utils import load_config
from .visualize import visualize
vis_show_cfg = load_config()["show"]

# ---------------------------------
# Batch Item
# ---------------------------------

@dataclass
class DetectorBatchItem:
    image: np.ndarray
    image_path: str
    foreground: ForegroundMaskBatchItem
    anomaly: AnomalyCLIPBatchItem
    anomaly_cls: ClassificationBatchItem
    segmentation: SegmentationBatchItem
    segmentation_cls: ClassificationBatchItem

    @property
    def regions(self):
        return self.anomaly.batch_regions

    def visualize(self) -> np.ndarray:
        return visualize(
            image=self.image,
            foreground=self.foreground,
            anomaly=self.anomaly,
            anomaly_cls=self.anomaly_cls,
            segmentation=self.segmentation,
            segmentation_cls=self.segmentation_cls,
            show_foreground=vis_show_cfg["foreground"],
            show_anomaly_map=vis_show_cfg["anomaly_map"],
            show_anomaly_score=vis_show_cfg["anomaly_score"],
        )


# ---------------------------------
# Output
# ---------------------------------

@dataclass
class DetectorOutput:
    images: Sequence[np.ndarray]
    images_path: Sequence[str]
    foreground: ForegroundMaskOutput
    anomaly: AnomalyCLIPOutput
    anomaly_cls: ClassificationBatchItem
    segmentation: SegmentationOutput
    segmentation_cls: ClassificationBatchItem

    def __post_init__(self):
        B = len(self.images)

        if len(self.foreground) != B:
            raise ValueError("Foreground batch size mismatch")

        if len(self.anomaly) != B:
            raise ValueError("Anomaly batch size mismatch")

        if len(self.anomaly_cls) != B:
            raise ValueError("Classification batch size mismatch")

        if len(self.segmentation) != B:
            raise ValueError("Segmentation batch size mismatch")

        if len(self.segmentation_cls) != B:
            raise ValueError("Segmentation classification batch size mismatch")

    def __len__(self):
        return len(self.images)

    def __iter__(self) -> Iterator[DetectorBatchItem]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> DetectorBatchItem:
        return DetectorBatchItem(
            image=self.images[idx],
            image_path=self.images_path[idx],
            foreground=self.foreground[idx],
            anomaly=self.anomaly[idx],
            anomaly_cls=self.anomaly_cls[idx],
            segmentation=self.segmentation[idx],
            segmentation_cls=self.segmentation_cls[idx],
        )