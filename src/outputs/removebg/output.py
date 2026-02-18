from typing import List
from dataclasses import dataclass
import numpy as np

@dataclass
class ForegroundMaskBatchItem:
    __slots__ = (
        "mask",
        "polygons_n",
    )

    mask: np.ndarray                 # (H, W)
    polygons_n: List[np.ndarray]     # (N, 2) float32 normalized

@dataclass
class ForegroundMaskOutput:
    __slots__ = (
        "masks",
        "polygons_n",
    )

    masks: np.ndarray                       # (B, H, W)
    polygons_n: List[List[np.ndarray]]        # per image polygons_n

    def __post_init__(self):
        if not isinstance(self.masks, np.ndarray):
            raise TypeError("masks must be np.ndarray")

        if self.masks.ndim != 3:
            raise ValueError("masks must have shape (B, H, W)")

        if len(self.polygons_n) != self.masks.shape[0]:
            raise ValueError("polygons_n batch size mismatch")

    def apply(self, array) -> np.ndarray:
        if isinstance(array, list):
            array = np.stack(array, axis=0)

        mask = self.masks
        if array.ndim == 4:
            mask = mask[..., None]

        return (mask * array).astype(np.uint8)

    def __len__(self):
        return self.masks.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield ForegroundMaskBatchItem(
                mask=self.masks[i],
                polygons_n=self.polygons_n[i],
            )

    def __getitem__(self, idx):
        return ForegroundMaskBatchItem(
            mask=self.masks[idx],
            polygons_n=self.polygons_n[idx],
        )
