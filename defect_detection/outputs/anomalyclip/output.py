from typing import List, Tuple
from dataclasses import dataclass, field
import cv2
import numpy as np


@dataclass
class AnomalyRegion:
    __slots__ = (
        "polygon",
        "polygon_n",
        "bboxes_xyxy",
        "bboxes_xyxy_n",
        "anomaly_score",
        "area",
        "area_n"
    )

    polygon: np.ndarray
    polygon_n: np.ndarray            # (N, 2) float32 normalized
    bboxes_xyxy: Tuple[int, int, int, int]
    bboxes_xyxy_n: Tuple[float, float, float, float]
    anomaly_score: float
    area: float
    area_n: float

@dataclass
class AnomalyCLIPBatchItem:
    __slots__ = (
        "map",
        "regions",
        "global_score",
    )
    map: np.ndarray
    regions: List[AnomalyRegion]
    global_score: float

@dataclass
class AnomalyCLIPOutput:
    __slots__ = (
        "maps",
        "score_threshold",
        "area_threshold",
        "batch_regions",
        "global_scores",
    )

    # declare only fields that should be declared as dataclass fields
    maps: np.ndarray                 # (B, H, W)
    score_threshold: float
    area_threshold: float

    def __post_init__(self):
        self._validate_inputs()

        batch_regions, global_scores = self._extract_regions_batch()

        # create slots-only attributes here
        object.__setattr__(self, "batch_regions", batch_regions)
        object.__setattr__(self, "global_scores", global_scores)

    def _validate_inputs(self):

        if not isinstance(self.maps, np.ndarray):
            raise TypeError("maps must be np.ndarray")

        if self.maps.ndim != 3:
            raise ValueError("maps must have shape (B, H, W)")

        if not isinstance(self.score_threshold, (float, int)):
            raise TypeError("score_threshold must be float")

        if not isinstance(self.area_threshold, (float, int)):
            raise TypeError("area_threshold must be float")

    def _extract_regions_batch(
        self,
    ) -> Tuple[List[List[AnomalyRegion]], List[float]]:

        regions_batch = []
        global_scores_batch = []

        for amap in self.maps:
            regions = self._extract_regions_single(amap)
            global_score = self._compute_global_score(amap)

            regions_batch.append(regions)
            global_scores_batch.append(global_score)

        return regions_batch, global_scores_batch

    def _extract_regions_single(
        self,
        amap: np.ndarray,
    ) -> List[AnomalyRegion]:

        H, W = amap.shape
        binary = (amap >= self.score_threshold).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        regions = []

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.area_threshold:
                continue
            area_n = area / float(H * W)

            polygon = cnt.reshape(-1, 2).astype(np.int32)

            polygon_n = polygon.astype(np.float32)
            polygon_n[:, 0] = np.clip(polygon_n[:, 0] / W, 0.0, 1.0)
            polygon_n[:, 1] = np.clip(polygon_n[:, 1] / H, 0.0, 1.0)

            x, y, w, h = cv2.boundingRect(cnt)
            bbox_xyxy = (x, y, x + w, y + h)

            bboxes_xyxy_n = (
                max(0.0, min(1.0, x / W)),
                max(0.0, min(1.0, y / H)),
                max(0.0, min(1.0, (x + w) / W)),
                max(0.0, min(1.0, (y + h) / H)),
            )

            score = self._compute_polygon_score(amap, polygon)

            regions.append(
                AnomalyRegion(
                    polygon=polygon,
                    polygon_n=polygon_n,
                    bboxes_xyxy=bbox_xyxy,
                    bboxes_xyxy_n=bboxes_xyxy_n,
                    anomaly_score=score,
                    area=area,
                    area_n=area_n,
                )
            )

        return regions

    def _compute_polygon_score(
        self,
        anomaly_map: np.ndarray,
        polygon: np.ndarray,
        mode: str = "mean",
    ) -> float:

        mask = np.zeros_like(anomaly_map, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)

        values = anomaly_map[mask == 1]

        if values.size == 0:
            return 0.0

        if mode == "mean":
            return float(values.mean())

        if mode == "max":
            return float(values.max())

        raise ValueError(mode)

    def _compute_global_score(
        self,
        anomaly_map: np.ndarray,
        mode: str = "topk",
        topk_ratio: float = 0.05,
    ) -> float:

        flat = anomaly_map.reshape(-1)

        if mode == "mean":
            return float(flat.mean())

        if mode == "max":
            return float(flat.max())

        if mode == "topk":
            k = max(1, int(len(flat) * topk_ratio))
            return float(np.partition(flat, -k)[-k:].mean())

        raise ValueError(mode)

    def __len__(self):
        return self.maps.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield AnomalyCLIPBatchItem(
                map=self.maps[i],
                regions=self.batch_regions[i],
                global_score=self.global_scores[i],
            )

    def __getitem__(self, idx):
        return AnomalyCLIPBatchItem(
            map=self.maps[idx],
            regions=self.batch_regions[idx],
            global_score=self.global_scores[idx],
        )
