import time
from typing import List

import cv2

from anomalyclip import AnomalyCLIPInference
from removebg import BackgroundRemover
from classify import Classifier

def inference(
    imgs_path: List[str],
    anomalyclip_checkpoint_path: str,
    bgremover_checkpoint_path: str,
    classifier_checkpoint_path: str,
    imgsz: int = 544,
    verbose: bool = True,
):

    # load models
    inferencer = AnomalyCLIPInference(
        checkpoint_path=anomalyclip_checkpoint_path,
        imgsz=imgsz,
    )
    bgremover = BackgroundRemover(
        checkpoint_path=bgremover_checkpoint_path,
        imgsz=imgsz,
    )
    classifier = Classifier(
        checkpoint_path=classifier_checkpoint_path,
        anomaly_threshold=0.25,
        min_area=10,
    )
    # load image
    t0 = time.time()
    imgs_np = [cv2.imread(img_path) for img_path in imgs_path]
    imgs_np = [cv2.resize(img, (imgsz, imgsz)) for img in imgs_np]

    
    # get anomaly map
    t1 = time.time()
    anomaly_maps, image_scores = inferencer.infer(imgs_np)

    # get foreground mask
    t2 = time.time()
    foreground_masks = bgremover.remove(imgs_np)

    # filter anomaly map by foreground mask
    t3 = time.time()
    filtered_anomaly_maps = anomaly_maps * foreground_masks

    # classify
    t4 = time.time()
    results = classifier.classify_batch(
        images=imgs_np,
        filtered_anomaly_maps=filtered_anomaly_maps,
    )

    t5 = time.time()
    if verbose:
        print(f"load image: {(t1 - t0)*1000:.2f}ms")
        print(f"get anomaly map: {(t2 - t1)*1000:.2f}ms")
        print(f"get foreground mask: {(t3 - t2)*1000:.2f}ms")
        print(f"filter anomaly map: {(t4 - t3)*1000:.2f}ms")
        print(f"classify: {(t5 - t4)*1000:.2f}ms")
        print(f"total: {(t5 - t0)*1000:.2f}ms")

    return results