import os
import tqdm
import torch
from model import Model

if __name__ == "__main__":

    model = Model(
        checkpoint_path="./checkpoints/AnomalyCLIPTrainSH_9_12_4_mvtec_anomaly_detection+F1038-F2150-M2520/epoch_15.pth",
        image_size=518,
    )
    
    model.warmup()

    src_dir = "./NAS/dataset_260204_F2150-M2520-F1038/dataset_final_for_anomaly_for_test/F2150"
    dst_dir = "./NAS/datasets_by_anomalyCLIP/AnomalyCLIPTrainSH_9_12_4_mvtec_anomaly_detection+F1038-F2150-M2520"
    for image in tqdm.tqdm(os.listdir(src_dir)):
        # if not image.endswith(".png") or not image.endswith(".jpg"):
        #     print(f"Skipping {image}")
        #     continue
        
        image_path = os.path.join(src_dir, image)
        model.pred(
            image_path=image_path,
            save_dir=dst_dir,
        )