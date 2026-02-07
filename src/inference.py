import cv2
from anomalyclip import AnomalyCLIPInference
from anomalyclip.utils import visualize
from removebg import BackgroundRemover

if __name__ == "__main__":

    imgsz = 544
    imgs_path = (
        "./tests/SSBR_1_20251227_000101_701.jpg",
        "./tests/SSBR_1_20260204_000027_617.jpg"
    )

    # load models
    inferencer = AnomalyCLIPInference(
        checkpoint_path="./checkpoints/9_12_4_mvtec+F1038-F2150-M2520/epoch_15.pth",
        features_list=[6, 12, 18, 24],
        imgsz=imgsz,
    )
    bgremover = BackgroundRemover(
        checkpoint_path="./NAS/segment/weights/rmbg/SSBR_F2150-M2520-F1038/weights/best.pt",
        imgsz=imgsz,
    )

    # load image
    imgs_np = [cv2.imread(img_path) for img_path in imgs_path]
    imgs_np = [cv2.resize(img, (imgsz, imgsz)) for img in imgs_np]

    # get anomaly map
    anomaly_maps, image_scores = inferencer.infer(imgs_np)

    # get foreground mask
    foreground_masks = bgremover.remove(imgs_np)

    # filter anomaly map by foreground mask
    filtered_anomaly_maps = anomaly_maps * foreground_masks

    # visualize filtered anomaly map
    for i, (img, amap, score) in enumerate(zip(imgs_np, filtered_anomaly_maps, image_scores)):
        img_path = imgs_path[i]
        visualize(
            img,
            amap[None],   # visualize가 (1, H, W) 기대하니까
            save_path=f"{img_path}_anomaly_map_{score:.4f}.jpg"
        )
        print(f"anomaly score: {img_path} {score:.4f}")
