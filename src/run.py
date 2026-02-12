from inspector import AnomalyInspector
from classify import visualize

if __name__ == "__main__":

    anomalyclip_checkpoint_path = "./NAS/anomaly/weights/SSBR/9_12_4_mvtec+F1038-F2150-M2520/epoch_15.pth"
    bgremover_checkpoint_path = "./NAS/segment/weights/rmbg/SSBR_F2150-M2520-F1038/weights/best.pt"
    classifier_checkpoint_path = "./NAS/classify/weights/SSBR/2550H_imgsz32/weights/best.pt"
    anomalyclip_imgsz = 32*12
    bgremover_imgsz = 32*5
    classifier_imgsz = 32
    anomaly_threshold = 0.25
    anomaly_min_area = 112
    classifier_conf_threshold = 0.5

    anomaly_inspector = AnomalyInspector(
        anomalyclip_checkpoint_path=anomalyclip_checkpoint_path,
        bgremover_checkpoint_path=bgremover_checkpoint_path,
        classifier_checkpoint_path=classifier_checkpoint_path,
        anomalyclip_imgsz=anomalyclip_imgsz,
        bgremover_imgsz=bgremover_imgsz,
        classifier_imgsz=classifier_imgsz,
        anomaly_threshold=anomaly_threshold,
        anomaly_min_area=anomaly_min_area,
        classifier_conf_threshold=classifier_conf_threshold,
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