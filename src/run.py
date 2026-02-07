from inference import inference
from classify import visualize

if __name__ == "__main__":

    imgsz = 32*12
    imgs_path = (
        "./tests/SSBR_1_20251227_000101_701.jpg",
        "./tests/SSBR_1_20260204_000027_617.jpg"
    )
    anomalyclip_checkpoint_path = "./NAS/anomaly/weights/SSBR/9_12_4_mvtec+F1038-F2150-M2520/epoch_15.pth"
    bgremover_checkpoint_path = "./NAS/segment/weights/rmbg/SSBR_F2150-M2520-F1038/weights/best.pt"
    classifier_checkpoint_path = "./NAS/classify/weights/SSBR/2550H_imgsz32/weights/best.pt"

    results = inference(
        imgs_path=imgs_path,
        anomalyclip_checkpoint_path=anomalyclip_checkpoint_path,
        bgremover_checkpoint_path=bgremover_checkpoint_path,
        classifier_checkpoint_path=classifier_checkpoint_path,
        imgsz=imgsz,
        verbose=True,
    )

    for i, res in enumerate(results):
        save_path = f"{imgs_path[i]}_classified.jpg"

        visualize(
            result=res,
            save_path=save_path,
            draw_anomaly_map=True
        )