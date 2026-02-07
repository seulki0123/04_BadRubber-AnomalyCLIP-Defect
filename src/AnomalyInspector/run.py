from inspector import AnomalyInspector
from classify import visualize

if __name__ == "__main__":

    anomalyclip_checkpoint_path = "./NAS/anomaly/weights/SSBR/9_12_4_mvtec+F1038-F2150-M2520/epoch_15.pth"
    bgremover_checkpoint_path = "./NAS/segment/weights/rmbg/SSBR_F2150-M2520-F1038/weights/best.pt"
    classifier_checkpoint_path = "./NAS/classify/weights/SSBR/2550H_imgsz32/weights/best.pt"
    imgsz = 32*12

    anomaly_inspector = AnomalyInspector(
        anomalyclip_checkpoint_path=anomalyclip_checkpoint_path,
        bgremover_checkpoint_path=bgremover_checkpoint_path,
        classifier_checkpoint_path=classifier_checkpoint_path,
        imgsz=imgsz,
    )

    imgs_path = (
        "./tests/SSBR_1_20251227_000101_701.jpg",
        "./tests/SSBR_1_20260204_000027_617.jpg"
    )

    # 1. inspect
    results = anomaly_inspector.inspect(imgs_path)

    for i, res in enumerate(results):
        save_path = f"{imgs_path[i]}_classified.jpg"

        visualize(
            result=res,
            save_path=save_path,
            draw_anomaly_map=True
        )

    # 2. detect faulty spots
    faulty_spots_results = anomaly_inspector.detect_faulty_spots_batch(imgs_path)
    print(faulty_spots_results)