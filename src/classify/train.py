import os
from ultralytics import YOLO
from ultralytics.data.split import split_classify_dataset

# split_classify_dataset("../datasets/LG_Chemistry_S1K2/data/annotations/SSBR_classify/251208", 0.8)

def main():
    model = YOLO("yolo11n-cls.pt")
    results = model.train(
        data="./NAS/classify/datasets/annotations/SSBR/2550H/01+02",
        project="./NAS/classify/weights/SSBR",
        name="2550H_imgsz32",
        epochs=30,
        imgsz=32,
        workers=8,

        # augment
        # hsv_h=0.0,
        # hsv_s=0.0,
        # hsv_v=0.0,
        # translate=0.0,
        # scale=0.0,
        # fliplr=0.0,
        # mosaic=0.0,
        erasing=0.0,
        # auto_augment=None,
    )

if __name__ == "__main__":
    main()