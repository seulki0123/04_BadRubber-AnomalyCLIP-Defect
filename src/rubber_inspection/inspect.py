import time
from typing import List, Tuple

import cv2

from models import AnomalyCLIPInference, BackgroundRemover, Classifier, RegionClassifierAdapter, SAM2Inference
from outputs import RegionClassificationOutput
from utils import load_config, random_color
from .result import InspectorOutput
from .visualize import draw_normalized_polygons

class Inspector:
    def __init__(self):
        config = load_config()

        self.imgsz = config["anomalyclip"]["imgsz"]

        self.anomaly_extractor = AnomalyCLIPInference(
            checkpoint_path=config["anomalyclip"]["checkpoint"],
            imgsz=self.imgsz,
            score_threshold=config["anomalyclip"]["threshold"],
            area_threshold=config["anomalyclip"]["min_area"],
        )

        self.bgremover = BackgroundRemover(
            checkpoint_path=config["bgremover"]["checkpoint"],
            imgsz=config["bgremover"]["imgsz"],
        )

        self.region_classifier = RegionClassifierAdapter(
            Classifier(
            checkpoint_path=config["classifier"]["checkpoint"],
            imgsz=config["classifier"]["imgsz"],
            conf_threshold=config["classifier"].get("threshold", 0.5),
            )
        )

        self.sam2 = SAM2Inference(
            checkpoint_path=config["sam2"]["checkpoint"],
            config_name=config["sam2"]["config_name"],
            pred_iou_thresh=config["sam2"]["pred_iou_thresh"],
            stability_score_thresh=config["sam2"]["stability_score_thresh"],
            points_per_side=config["sam2"]["points_per_side"],
        )

    # ---------------------------------
    # Main API
    # ---------------------------------

    def inspect(self, imgs_path: List[str]) -> InspectorOutput:
        t0 = time.time()

        images = [cv2.imread(p) for p in imgs_path]
        images = [cv2.resize(img, (self.imgsz, self.imgsz)) for img in images]
        t1 = time.time()

        foreground = self.bgremover.infer(images)
        t2 = time.time()

        anomaly = self.anomaly_extractor.infer(images, foreground.masks)
        t3 = time.time()
        
        classification = self.region_classifier.classify(images, anomaly)
        t4 = time.time()

        sam2_output = self.sam2.infer(images, anomaly.regions, visualize=True)
        t5 = time.time()

        print(f"load images: {(t1-t0)*1000}ms")
        print(f"foreground: {(t2-t1)*1000}ms")
        print(f"anomaly: {(t3-t2)*1000}ms")
        print(f"classification: {(t4-t3)*1000}ms")
        print(f"SAM2: {(t5-t4)*1000}ms")
        print(f"image count: {len(images)}")
        print(f"total: {(t5-t0)*1000}ms")
        print(f"time per image: {(t5-t0)*1000/len(images)}ms")

        return InspectorOutput(
            images=images,
            images_path=imgs_path,
            foreground=foreground,
            anomaly=anomaly,
            classification=classification,
        )