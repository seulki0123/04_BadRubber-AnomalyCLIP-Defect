import os
import argparse

import cv2
import tqdm

from inspector import AnomalyInspector
from classify import visualize
from utils import Report

def batch(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def run(src_root, dst_root, line, grade, dates, batch_size=9):
    anomaly_inspector = AnomalyInspector()
    report = Report()

    # dates
    for date in dates:
        src_dir = os.path.join(src_root, line, date)
        dst_dir = os.path.join(dst_root, line, grade, date)

        dst_segmentation_dir = os.path.join(dst_dir, "segmentation")
        dst_crops_dir = os.path.join(dst_dir, "crops")
        os.makedirs(dst_segmentation_dir, exist_ok=True)
        os.makedirs(dst_crops_dir, exist_ok=True)

        cams = [
            d for d in tqdm.tqdm(os.listdir(src_dir), desc=f"Loading {date} {line} {grade} cams")
            if os.path.isdir(os.path.join(src_dir, d))
            and d.startswith("CAM")
        ] or [""]

        # cams
        for cam in cams:
            cam_dir = os.path.join(src_dir, cam)

            files = [f for f in os.listdir(cam_dir)if f.endswith(".jpg")]

            # batch inference
            for file_batch in tqdm.tqdm(list(batch(files, batch_size=9)),desc=f"{date} {cam} batch inference"):
                print("\nbatch size: ", len(file_batch))
                img_paths = [os.path.join(cam_dir, f) for f in file_batch]
                batch_results = anomaly_inspector.inspect(img_paths, verbose=True)

                # result
                for file, res in zip(file_batch, batch_results):
                    vis_img, crops_img = visualize(result=res, draw_anomaly_map=False, crop_scale=2.0)
                    cv2.imwrite(os.path.join(dst_segmentation_dir, f"{file}_inspect-result.jpg"), vis_img)

                    for idx, crop in enumerate(crops_img):
                        save_name = (
                            f"{crop['cls_name']}_{file}_{idx:02d}_"
                            f"{crop['conf']:.2f}_A{crop['a_score']:.2f}.jpg"
                        )
                        cv2.imwrite(
                            os.path.join(dst_crops_dir, save_name),
                            crop["img"]
                        )

                    report.update([i["class_name"] for i in res["regions"] if not i["pass"]])
                
                report.print_report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=str, default="./NAS/LG_Chemistry_S1K2")
    parser.add_argument("--dst-root", type=str, default="./NAS/LG_Chemistry_Site")
    parser.add_argument("--line", type=str)
    parser.add_argument("--grade", type=str)
    parser.add_argument("--dates", type=str, nargs="+")
    parser.add_argument("--batch-size", type=int, default=9, help="Batch size")
    args = parser.parse_args()

    run(src_root=args.src_root, dst_root=args.dst_root, line=args.line, grade=args.grade, dates=args.dates, batch_size=args.batch_size)