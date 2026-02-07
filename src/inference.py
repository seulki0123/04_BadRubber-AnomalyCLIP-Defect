import cv2
from anomalyclip import AnomalyCLIPInference
from anomalyclip.utils import visualize
from removebg import BackgroundRemover

if __name__ == "__main__":

    imgsz = 544
    image_path = "./tests/SSBR_1_20251227_000101_701.jpg"

    # load models
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
    bgremover = BackgroundRemover(
        checkpoint_path="./NAS/segment/weights/rmbg/SSBR_F2150-M2520-F1038/weights/best.pt",
        imgsz=imgsz,
    )

    # load image
    img_np = cv2.imread(image_path)
    img_np = cv2.resize(img_np, (imgsz, imgsz))

    # get anomaly map
    anomaly_map, image_score = inferencer.infer(img_np)

    # get foreground mask
    foreground_mask = bgremover.remove(img_np)

    # filter anomaly map by foreground mask
    filtered_anomaly_map = anomaly_map * foreground_mask

    # visualize filtered anomaly map
    visualize(img_np, filtered_anomaly_map, save_path=image_path+"_anomaly_map.jpg")
    print(anomaly_map)
    print(foreground_mask)
    print(filtered_anomaly_map)
    print(image_score)