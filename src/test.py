import os
import tqdm
import torch
from model import Model

def run(src_dir, dst_dir, anomalyclip_weights, segmentation_weights, classification_weights, image_size=518, save_anomaly_map=True, save_segmentation=True, save_crops=True):
    model = Model(
        anomalyclip_weights=anomalyclip_weights,
        segmentation_weights=segmentation_weights,
        classification_weights=classification_weights,
        image_size=image_size,
        save_anomaly_map=save_anomaly_map,
        save_segmentation=save_segmentation,
        save_crops=save_crops,
    )
    
    model.warmup()

    TotalEA = 0

    적합EA = 0
    부적합EA = 0

    전체검출수EA = 0
    Bale평균검출수EA = 0

    적합per = 0
    부적합per = 0

    Class검출수 = {}

    def pinrt2():
        print(f"적합EA: {적합EA}")
        print(f"부적합EA: {부적합EA}")
        print(f"전체검출수EA: {전체검출수EA}")
        print(f"Bale평균검출수EA: {Bale평균검출수EA}")
        print(f"적합per: {적합per:.2f}")
        print(f"부적합per: {부적합per:.2f}")
        print(f"Class검출수: {Class검출수}")
        print("-"*100, "\n")

    idx = 0
    for image in tqdm.tqdm(os.listdir(src_dir)):
        if not image.endswith(".jpg"):
            continue

        image_path = os.path.join(src_dir, image)
        labels = model.pred(
            image_path=image_path,
            save_dir=dst_dir,
        )

        TotalEA += 1
        print(labels)
        if len(labels):
            부적합EA += 1
        else:
            적합EA += 1

        전체검출수EA += len(labels)
        Bale평균검출수EA = 전체검출수EA / TotalEA

        적합per = 적합EA / TotalEA
        부적합per = 부적합EA / TotalEA

        for label in labels:
            Class검출수[label] = Class검출수.get(label, 0) + 1

        if idx % 100 == 0 or idx == len(os.listdir(src_dir)) - 1:
            pinrt2()

        idx += 1
    
if __name__ == "__main__":

    date = "2026-01-04"
    grade = "M2520"
    src_dir = f"./NAS/백업/LG_Chemistry_Site/SSBR/{date}"
    dst_dir = f"./NAS/defect/classify/datasets/raw/{grade}/{date}"

    anomalyclip_weights = "./NAS/defect/anomaly/weights/SSBR/9_12_4_mvtec+F1038-F2150-M2520/epoch_15.pth"
    segmentation_weights = "./NAS/defect/segment/weights/rmbg/SSBR_F2150-M2520-F1038/weights/best.pt"
    classification_weights = "./NAS/defect/classify/weights/SSBR/F2150/weights/best.pt"

    run(
        src_dir=src_dir,
        dst_dir=dst_dir,
        anomalyclip_weights=anomalyclip_weights,
        segmentation_weights=segmentation_weights,
        classification_weights=classification_weights,
        save_anomaly_map=True,
        save_segmentation=True,
        save_crops=True,
    )