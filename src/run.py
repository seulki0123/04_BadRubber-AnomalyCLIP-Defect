from inspector import AnomalyInspector
from classify import visualize

if __name__ == "__main__":
    anomaly_inspector = AnomalyInspector()

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