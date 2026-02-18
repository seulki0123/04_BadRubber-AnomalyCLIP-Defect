import cv2
from typing import List, Tuple

from models import AnomalyCLIPInference, BackgroundRemover, Classifier, RegionClassifierAdapter
from outputs import RegionClassificationOutput
from utils import load_config
from .result import InspectorOutput


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

    # ---------------------------------
    # Main API
    # ---------------------------------

    def inspect(self, imgs_path: List[str]) -> InspectorOutput:

        images = [cv2.imread(p) for p in imgs_path]
        images = [cv2.resize(img, (self.imgsz, self.imgsz)) for img in images]

        foreground = self.bgremover.infer(images)
        anomaly = self.anomaly_extractor.infer(images, foreground.masks)

        classification = self.region_classifier.classify(images, anomaly)

        return InspectorOutput(
            images=images,
            images_path=imgs_path,
            foreground=foreground,
            anomaly=anomaly,
            classification=classification,
        )