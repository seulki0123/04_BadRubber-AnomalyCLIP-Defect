from dataclasses import dataclass
from typing import List, Iterator, Tuple, Any
import numpy as np


@dataclass(frozen=True)
class Segmentation:
    class_id: int
    class_name: str
    confidence: float
    polygon_n: np.ndarray
    color: Tuple[int, int, int]


@dataclass(frozen=True)
class SegmentationBatchItem:
    regions: List[Segmentation]

    def __len__(self):
        return len(self.regions)

    def __iter__(self):
        return iter(self.regions)


@dataclass
class RegionSegmentationOutput:
    __slots__ = ("batch",)

    batch: List[List[Segmentation]]  # [B][R]

    def __len__(self):
        return len(self.batch)

    def __iter__(self) -> Iterator[SegmentationBatchItem]:
        for regions in self.batch:
            yield SegmentationBatchItem(regions)

    def __getitem__(self, idx: int) -> SegmentationBatchItem:
        return SegmentationBatchItem(self.batch[idx])
