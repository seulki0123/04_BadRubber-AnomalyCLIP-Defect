import time
from typing import List, Tuple, Optional, Any, Dict

import cv2

from .anomalyclip import AnomalyCLIPInference
from .removebg import BackgroundRemover
from .classify import Classifier, visualize


class AnomalyInspector:
    def __init__(
        self,
        anomalyclip_checkpoint_path: str,
        bgremover_checkpoint_path: str,
        classifier_checkpoint_path: str,
        anomalyclip_imgsz: int = 32 * 8,
        bgremover_imgsz: int = 32 * 5,
        classifier_imgsz: int = 32,
        anomaly_threshold: float = 0.25,
        anomaly_min_area: int = 112,
        classifier_conf_threshold: float = 0.5,
    ):
        self.imgsz = anomalyclip_imgsz

        self.inferencer = AnomalyCLIPInference(
            checkpoint_path=anomalyclip_checkpoint_path,
            imgsz=anomalyclip_imgsz,
        )

        self.bgremover = BackgroundRemover(
            checkpoint_path=bgremover_checkpoint_path,
            imgsz=bgremover_imgsz,
        )

        self.classifier = Classifier(
            checkpoint_path=classifier_checkpoint_path,
            anomaly_threshold=anomaly_threshold,
            min_area=anomaly_min_area,
            conf_threshold=classifier_conf_threshold,
            imgsz=classifier_imgsz,
        )

    def inspect(self, imgs_path: List[str], verbose: bool = True) -> List[Dict[str, Any]]:
        t0 = time.time()

        # load image
        imgs_np = [cv2.imread(p) for p in imgs_path]
        orig_sizes = [img.shape[:2] for img in imgs_np]
        imgs_np = [cv2.resize(img, (self.imgsz, self.imgsz)) for img in imgs_np]

        t1 = time.time()

        # anomaly map
        anomaly_maps, image_scores = self.inferencer.infer(imgs_np)
        t2 = time.time()

        # foreground mask
        foreground_masks = self.bgremover.remove(imgs_np)
        t3 = time.time()

        # filter anomaly map
        filtered_anomaly_maps = anomaly_maps * foreground_masks
        t4 = time.time()

        # classify
        results = self.classifier.classify_batch(
            images=imgs_np,
            filtered_anomaly_maps=filtered_anomaly_maps,
        )
        for i, res in enumerate(results):
            res["orig_size"] = orig_sizes[i]  # (h, w)
        t5 = time.time()

        if verbose:
            print(f"load image: {(t1 - t0)*1000:.2f}ms")
            print(f"get anomaly map: {(t2 - t1)*1000:.2f}ms")
            print(f"get foreground mask: {(t3 - t2)*1000:.2f}ms")
            print(f"filter anomaly map: {(t4 - t3)*1000:.2f}ms")
            print(f"classify: {(t5 - t4)*1000:.2f}ms")
            print(f"total: {(t5 - t0)*1000:.2f}ms")

        return results

    def detect_faulty_spots_batch(self, imgs_path: List[str], verbose: bool = True) -> List[Dict[str, Any]]:
        """
            results = [
                (
                    vis_img,
                    [{"class": int, "class_name": str, "confidence": float, "bbox": [x1, y1, x2, y2]}],
                    False,
                ), ...
            ]
        """
        r = self.inspect(imgs_path, verbose)

        batch_results = []
        for i, res in enumerate(r):
            # original size
            h0, w0 = res["orig_size"]

            # detections
            dets = []
            for region in res["regions"]:
                if region["pass"]:
                    continue

                x1n, y1n, x2n, y2n = region["bbox_n"]
                bbox_orig = [int(x1n * w0), int(y1n * h0), int(x2n * w0), int(y2n * h0)]
                
                dets.append({
                    "class": int(region["class_id"]),
                    "class_name": region["class_name"],
                    "confidence": region["confidence"],
                    "bbox": bbox_orig,  # (x1, y1, x2, y2)
                })

            # has_faulty
            has_faulty = len(dets) > 0

            # annotated image
            ann_img = visualize(
                result=res,
                draw_anomaly_map=False
            ) if has_faulty else None

            batch_results.append((ann_img, dets, has_faulty))

        return batch_results