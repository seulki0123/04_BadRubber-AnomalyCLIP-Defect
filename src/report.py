import os
import argparse

import cv2
import tqdm

from utils import Report
from rubber_inspection import Inspector, crop_regions

def batch(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def run(src_root, dst_root, line, grade, dates, batch_size=9):
    report = Report()
    inspector = Inspector()

    # dates
    for date in dates:
        src_dir = os.path.join(src_root, line, date)
        dst_dir = os.path.join(dst_root, line, grade, date)

        dst_segmentation_dir = os.path.join(dst_dir, "segmentation")
        dst_crops_anomaly_dir = os.path.join(dst_dir, "crops_anomaly")
        dst_crops_segmentation_dir = os.path.join(dst_dir, "crops_segmentation")
        os.makedirs(dst_segmentation_dir, exist_ok=True)
        os.makedirs(dst_crops_anomaly_dir, exist_ok=True)
        os.makedirs(dst_crops_segmentation_dir, exist_ok=True)

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
                results = inspector.inspect(img_paths)

                for result in results:
                    # result image
                    imagename = os.path.basename(result.image_path)
                    cv2.imwrite(os.path.join(dst_segmentation_dir, f"{imagename}_inspect-result.jpg"), result.visualize())
                    
                    # anomaly crop images
                    anomaly_crops = crop_regions(result.image, result.anomaly.regions, result.anomaly_cls.regions, prefix="anomaly", name_from="cls")
                    for filename, crop in anomaly_crops.items():
                        cv2.imwrite(os.path.join(dst_crops_anomaly_dir, f"{filename}_{imagename}"), crop)

                    anomaly_crops = crop_regions(result.image, result.anomaly.regions, result.anomaly_cls.regions, prefix="anomaly", name_from="cls", draw_polygon=True)
                    for filename, crop in anomaly_crops.items():
                        cv2.imwrite(os.path.join(dst_crops_anomaly_dir, f"{filename}_{imagename}_polygon.jpg"), crop)
                    
                    # segmentation crop images
                    segmentation_crops = crop_regions(result.image, result.segmentation.regions, result.segmentation_cls.regions, prefix="segment", name_from="region")
                    for filename, crop in segmentation_crops.items():
                        cv2.imwrite(os.path.join(dst_crops_segmentation_dir, f"{filename}_{imagename}"), crop)

                    segmentation_crops = crop_regions(result.image, result.segmentation.regions, result.segmentation_cls.regions, prefix="segment", name_from="region", draw_polygon=True)
                    for filename, crop in segmentation_crops.items():
                        cv2.imwrite(os.path.join(dst_crops_segmentation_dir, f"{filename}_{imagename}_polygon.jpg"), crop)
                        
                    # report.update([i["class_name"] for i in res["regions"] if not i["pass"]])
                
                # report.print_report()

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