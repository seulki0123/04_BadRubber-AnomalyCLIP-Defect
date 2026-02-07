from typing import List, Tuple, Dict, Any, Sequence, Optional

import cv2
import numpy as np
from ultralytics import YOLO

BBox = Tuple[int, int, int, int]   # x1, y1, x2, y2
Polygon = np.ndarray               # (N, 2)

class Classifier:
    def __init__(
        self,
        checkpoint_path: str,
        anomaly_threshold: float = 0.5,
        min_area: int = 100,
    ) -> None:
        self.model = YOLO(checkpoint_path)
        self.anomaly_threshold = anomaly_threshold
        self.min_area = min_area

    def _extract_regions_batch(
        self,
        anomaly_maps: np.ndarray,   # (B, H, W)
    ) -> List[Dict[str, Any]]:
        outputs = []

        for amap in anomaly_maps:
            binary = (amap >= self.anomaly_threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            regions = []

            for cnt in contours:
                if cv2.contourArea(cnt) < self.min_area:
                    continue

                polygon = cnt.squeeze(1)
                x, y, w, h = cv2.boundingRect(cnt)
                bbox = (x, y, x + w, y + h)

                score = self.compute_polygon_score(
                    amap,
                    polygon,
                    mode="mean",
                )

                regions.append({
                    "polygon": polygon,
                    "bbox": bbox,
                    "anomaly_score": score,
                })

            global_score = self.compute_global_score(amap)

            outputs.append({
                "regions": regions,
                "global_anomaly_score": global_score,
            })

        return outputs

    def _classify_crops_batch(
        self,
        images: Sequence[np.ndarray],
        bboxes_batch: List[List[BBox]],
    ) -> List[List[Dict[str, Any]]]:
        """
        Returns:
            detections_batch[B][N]
        """
        crops: List[np.ndarray] = []
        crop_to_image: List[Tuple[int, BBox]] = []

        # flatten crops
        for img_idx, (img, bboxes) in enumerate(zip(images, bboxes_batch)):
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                crop = img[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                crops.append(crop)
                crop_to_image.append((img_idx, bbox))

        # no regions at all
        if not crops:
            return [[] for _ in images]

        # YOLOv11 batch classification
        preds = self.model(crops, verbose=False, imgsz=32)

        results_batch: List[List[Dict[str, Any]]] = [
            [] for _ in images
        ]

        for pred, (img_idx, bbox) in zip(preds, crop_to_image):
            if pred.probs is None:
                continue

            cls_id = int(pred.probs.top1)
            conf = float(pred.probs.top1conf)
            cls_name = self.model.names[cls_id]

            results_batch[img_idx].append({
                "bbox": bbox,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
            })

        return results_batch

    def classify_batch(
        self,
        images: Sequence[np.ndarray],
        filtered_anomaly_maps: np.ndarray,
    ) -> List[Dict[str, Any]]:

        regions_batch = self._extract_regions_batch(filtered_anomaly_maps)

        bboxes_batch = [
            [r["bbox"] for r in img_res["regions"]]
            for img_res in regions_batch
        ]

        detections_batch = self._classify_crops_batch(images, bboxes_batch)

        for img_idx, dets in enumerate(detections_batch):
            regions_batch[img_idx]["image"] = images[img_idx]
            regions_batch[img_idx]["anomaly_map"] = filtered_anomaly_maps[img_idx]

            for det in dets:
                for region in regions_batch[img_idx]["regions"]:
                    region.setdefault("class_name", None)
                    region.setdefault("confidence", None)
                    if tuple(region["bbox"]) == tuple(det["bbox"]):
                        region.update(det)
                        break

        return regions_batch


    def compute_global_score(
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
            return float(np.sort(flat)[-k:].mean())

        raise ValueError(f"Unknown mode: {mode}")


    def compute_polygon_score(
        self,
        anomaly_map: np.ndarray,
        polygon: np.ndarray,
        mode: str = "mean",
    ) -> float:
        """
        polygon: (N, 2)
        """
        mask = np.zeros_like(anomaly_map, dtype=np.uint8)

        cv2.fillPoly(
            mask,
            [polygon.astype(np.int32)],
            1,
        )

        values = anomaly_map[mask == 1]

        if values.size == 0:
            return 0.0

        if mode == "mean":
            return float(values.mean())

        if mode == "max":
            return float(values.max())

        raise ValueError(f"Unknown mode: {mode}")