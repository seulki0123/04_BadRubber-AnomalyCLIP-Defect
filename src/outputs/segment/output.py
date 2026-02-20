from dataclasses import dataclass
from typing import List, Iterator, Tuple, Any
import numpy as np


@dataclass
class Segmentation:
    __slots__ = (
        "polygon_n",
        "bboxes_xyxy_n",
        "confidence",
        "area",
        "class_id",
        "class_name",
        "color",
    )

    polygon_n: np.ndarray
    bboxes_xyxy_n: Tuple[float, float, float, float]
    confidence: float
    area: float
    class_id: int
    class_name: str
    color: Tuple[int, int, int]

@dataclass
class SegmentationBatchItem:
    __slots__ = ("regions",)

    regions: List[Segmentation]

    def __len__(self):
        return len(self.regions)

    def __iter__(self):
        return iter(self.regions)

    def __getitem__(self, idx: int) -> Segmentation:
        return self.regions[idx]

@dataclass
class SegmentationOutput:
    __slots__ = ("batch_regions",)

    batch_regions: List[List[Segmentation]]  # [B][R]

    def __post_init__(self):
        self._validate_inputs()

    def _validate_inputs(self):
        if not isinstance(self.batch_regions, list):
            raise TypeError("batch must be List[List[Segmentation]]")

        for regions in self.batch_regions:
            if not isinstance(regions, list):
                raise TypeError("Each batch item must be List[Segmentation]")
            for r in regions:
                if not isinstance(r, Segmentation):
                    raise TypeError("Elements must be Segmentation")

    def __len__(self):
        return len(self.batch_regions)

    def __iter__(self) -> Iterator[SegmentationBatchItem]:
        for regions in self.batch_regions:
            yield SegmentationBatchItem(regions=regions)

    def __getitem__(self, idx: int) -> SegmentationBatchItem:
        return SegmentationBatchItem(regions=self.batch_regions[idx])