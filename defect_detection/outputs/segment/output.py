from dataclasses import dataclass
from typing import List, Iterator, Tuple, Any
import numpy as np


@dataclass
class Segmentation:
    __slots__ = (
        "polygon",
        "polygon_n",
        "bboxes_xyxy",
        "bboxes_xyxy_n",
        "confidence",
        "area",
        "area_n",
        "class_id",
        "class_name",
        "color",
    )

    polygon: np.ndarray
    polygon_n: np.ndarray
    bboxes_xyxy: Tuple[int, int, int, int]
    bboxes_xyxy_n: Tuple[float, float, float, float]
    confidence: float
    area: float
    area_n: float
    class_id: int
    class_name: str
    color: Tuple[int, int, int]

@dataclass
class SegmentationBatchItem:
    __slots__ = ("regions",)

    # regions = [ [S], [S], ... ]  (R개)
    regions: List[List[Segmentation]]

    def __len__(self):
        return len(self.regions)  # R

    def __iter__(self):
        return iter(self.regions)

    def __getitem__(self, idx: int) -> List[Segmentation]:
        return self.regions[idx]


@dataclass
class SegmentationOutput:
    __slots__ = ("batch_regions",)

    # [B][R][S]
    batch_regions: List[List[List[Segmentation]]]

    def __post_init__(self):
        if not isinstance(self.batch_regions, list):
            raise TypeError("batch_regions must be List[List[List[Segmentation]]]")

    def __len__(self):
        return len(self.batch_regions)  # B

    def __iter__(self) -> Iterator[SegmentationBatchItem]:
        for regions in self.batch_regions:
            yield SegmentationBatchItem(regions=regions)

    def __getitem__(self, idx: int) -> SegmentationBatchItem:
        return SegmentationBatchItem(regions=self.batch_regions[idx])