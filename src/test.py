import os
import tqdm
import torch
from model import Model

if __name__ == "__main__":

    model = Model(
        anomalyclip_weights = "./NAS/anomaly/weights/SSBR/9_12_4_mvtec+F1038-F2150-M2520/epoch_15.pth",
        segmentation_weights = "./NAS/segment/weights/rmbg/SSBR_F2150-M2520-F1038/weights/best.pt",
        classification_weights = "./NAS/classify/weights/SSBR/F2150/weights/best.pt",
        image_size=518,
    )
    
    date = "2026-02-05"
    grade = "F2743"

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

    model.warmup()
    for cam in ["CAM1", "CAM2", "CAM3", "CAM4", "CAM5", "CAM6"]:

        src_dir = f"./NAS_Site_SSBR/SSBR/{date}/{cam}"
        dst_dir = f"./NAS/_report/SSBR/{grade}/{date}/{cam}"

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

            if idx % 100 == 0:
                pinrt2()

            idx += 1
    
        pinrt2()

    pinrt2()