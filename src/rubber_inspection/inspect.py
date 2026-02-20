import time
from typing import List, Tuple

import cv2

from models import AnomalyCLIPInference, BackgroundRemover, Classifier, RegionClassifierAdapter, Segmenter, RegionSegmenterAdapter
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
            conf_threshold=config["classifier"]["threshold"],
            )
        )

        self.region_segmenter = RegionSegmenterAdapter(
            Segmenter(
            checkpoint_path=config["segmenter"]["checkpoint"],
            imgsz=config["segmenter"]["imgsz"],
            conf_threshold=config["segmenter"]["threshold"],
            )
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
        
        anomaly_cls = self.region_classifier.infer(images, anomaly)
        t4 = time.time()

        segmentation = self.region_segmenter.infer(images, anomaly, anomaly_cls)
        t5 = time.time()

        segmentation_cls = self.region_classifier.infer(images, segmentation)
        t6 = time.time()

        print(f"load images: {(t1-t0)*1000}ms")
        print(f"foreground: {(t2-t1)*1000}ms")
        print(f"anomaly: {(t3-t2)*1000}ms")
        print(f"anomaly_cls: {(t4-t3)*1000}ms")
        print(f"segmentation: {(t5-t4)*1000}ms")
        print(f"segmentation_cls: {(t6-t5)*1000}ms")
        print(f"image count: {len(images)}")
        print(f"total: {(t6-t0)*1000}ms")
        print(f"time per image: {(t6-t0)*1000/len(images)}ms")

        return InspectorOutput(
            images=images,
            images_path=imgs_path,
            foreground=foreground,
            anomaly=anomaly,
            anomaly_cls=anomaly_cls,
            segmentation=segmentation,
            segmentation_cls=segmentation_cls,
        )