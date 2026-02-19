from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class SAM2Region:
    __slots__ = (
        "polygon_n",
        "area",
        "predicted_iou",
        "stability_score",
    )
    polygon_n: np.ndarray   # (N, 2)  [[x1,y1], [x2,y2], ...]
    area: float
    predicted_iou: float
    stability_score: float
    
@dataclass
class SAM2BatchItem:
    __slots__ = (
        "regions",
    )
    regions: List[SAM2Region]

@dataclass
class SAM2Output:
    __slots__ = ("batch",)

    batch: List[List[List[SAM2Region]]]
    # batch[b][r] -> List[SAM2Region]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, idx):
        return self.batch[idx]
