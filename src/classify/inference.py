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
        verbose: bool = True,
    ) -> None:
        self.model = YOLO(checkpoint_path)
        self.anomaly_threshold = anomaly_threshold
        self.min_area = min_area
        self.verbose = verbose

    def _extract_regions_batch(
        self,
        anomaly_maps: np.ndarray,   # (B, H, W)
    ) -> Tuple[
        List[List[Polygon]],
        List[List[BBox]],
    ]:
        all_polygons: List[List[Polygon]] = []
        all_bboxes: List[List[BBox]] = []

        for amap in anomaly_maps:
            binary = (amap >= self.anomaly_threshold).astype(np.uint8) * 255

            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            polygons: List[Polygon] = []
            bboxes: List[BBox] = []

            for cnt in contours:
                if cv2.contourArea(cnt) < self.min_area:
                    continue

                polygon = cnt.squeeze(1)
                x, y, w, h = cv2.boundingRect(cnt)

                polygons.append(polygon)
                bboxes.append((x, y, x + w, y + h))

            all_polygons.append(polygons)
            all_bboxes.append(bboxes)

        return all_polygons, all_bboxes

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
        preds = self.model(crops, verbose=self.verbose, imgsz=32)

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
        images: Sequence[np.ndarray],          # (B, H, W, 3)
        filtered_anomaly_maps: np.ndarray,     # (B, H, W)
    ) -> List[Dict[str, Any]]:
        polygons_batch, bboxes_batch = self._extract_regions_batch(
            filtered_anomaly_maps
        )

        detections_batch = self._classify_crops_batch(
            images, bboxes_batch
        )

        outputs: List[Dict[str, Any]] = []

        for i in range(len(images)):
            outputs.append({
                "image_index": i,
                "polygons": polygons_batch[i],
                "detections": detections_batch[i],
            })

        return outputs

    def visualize(
        self,
        image: np.ndarray,              # (H, W, 3)
        anomaly_map: np.ndarray,        # (H, W)
        result: Dict[str, Any],         # classify_batch[i]
        save_path: Optional[str] = None,
        alpha: float = 0.5,
        draw_anomaly_map: bool = False,
    ) -> np.ndarray:
        """
        Returns:
            vis_img: np.ndarray (H, W, 3)
        """
        vis_img = image.copy()

        if draw_anomaly_map:
            # --- anomaly heatmap overlay ---
            heatmap = (anomaly_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            vis_img = cv2.addWeighted(vis_img, 1 - alpha, heatmap, alpha, 0)

        # --- polygons + anomaly score ---
        for poly in result["polygons"]:
            poly_int = poly.astype(np.int32)

            score = self.compute_polygon_score(
                anomaly_map,
                poly,
                mode="mean",
            )

            cv2.polylines(
                vis_img,
                [poly_int],
                isClosed=True,
                color=(0, 255, 255),
                thickness=1,
            )

            # polygon 중심
            cx = int(poly_int[:, 0].mean())
            cy = int(poly_int[:, 1].mean())

            cv2.putText(
                vis_img,
                f"A:{score:.2f}",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

        # --- detections ---
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            conf = det["confidence"]

            cv2.rectangle(
                vis_img,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                1,
            )

            label = f"{cls_name} {conf:.2f}"
            cv2.putText(
                vis_img,
                label,
                (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        global_score = self.compute_global_score(
            anomaly_map,
            mode="topk",
            topk_ratio=0.05,
        )

        cv2.putText(
            vis_img,
            f"Global A: {global_score:.3f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if save_path is not None:
            cv2.imwrite(save_path, vis_img)

        return vis_img

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
