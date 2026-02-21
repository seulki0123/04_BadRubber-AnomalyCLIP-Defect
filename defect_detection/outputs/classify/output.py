from dataclasses import dataclass
from typing import List, Iterator, Tuple

# ---------------------------------
# Single region classification
# ---------------------------------

@dataclass(frozen=True)
class Classification:
    class_id: int
    class_name: str
    confidence: float
    is_pass: bool
    color: Tuple[int, int, int]

# ---------------------------------
# Per-image batch item
# ---------------------------------

@dataclass(frozen=True)
class ClassificationBatchItem:
    regions: List[Classification]

    def __len__(self):
        return len(self.regions)

    def __iter__(self):
        return iter(self.regions)


# ---------------------------------
# Batch output
# ---------------------------------

@dataclass
class RegionClassificationOutput:
    __slots__ = ("batch",)

    batch: List[List[Classification]]  # [B][R]

    def __len__(self):
        return len(self.batch)

    def __iter__(self) -> Iterator[ClassificationBatchItem]:
        for regions in self.batch:
            yield ClassificationBatchItem(regions)

    def __getitem__(self, idx: int) -> ClassificationBatchItem:
        return ClassificationBatchItem(self.batch[idx])
