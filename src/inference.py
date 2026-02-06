import cv2
from anomalyclip import AnomalyCLIPInference
from anomalyclip.inference import visualize

if __name__ == "__main__":

    imgsz = 518
    image_path = "./tests/SSBR_1_20251227_000101_701.jpg"

    inferencer = AnomalyCLIPInference(
        checkpoint_path="./checkpoints/9_12_4_mvtec+F1038-F2150-M2520/epoch_15.pth",
        features_list=[6, 12, 18, 24],
        imgsz=imgsz,
        depth=9,
        n_ctx=12,
        t_n_ctx=4,
        feature_map_layer=(0, 1, 2, 3),
        sigma=4,
    )

    img_np = cv2.imread(image_path)
    img_np = cv2.resize(img_np, (imgsz, imgsz))    

    anomaly_map, image_score = inferencer.infer(img_np)
    visualize(img_np, anomaly_map, save_path=image_path+"_anomaly_map.jpg")

    print(anomaly_map)
    print(image_score)