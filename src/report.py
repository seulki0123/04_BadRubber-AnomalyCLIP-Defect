import os
import json
import argparse

import cv2
import tqdm

from utils import Report, save_polygons_to_yolo_format
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

        dst_result_image_dir = os.path.join(dst_dir, "results", "images")
        dst_result_meta_dir = os.path.join(dst_dir, "results", "metadatas")
        dst_crop_images_dir = os.path.join(dst_dir, "crops", "images")
        dst_crop_labels_dir = os.path.join(dst_dir, "crops", "labels")
        os.makedirs(dst_result_image_dir, exist_ok=True)
        os.makedirs(dst_result_meta_dir, exist_ok=True)
        os.makedirs(dst_crop_images_dir, exist_ok=True)
        os.makedirs(dst_crop_labels_dir, exist_ok=True)
        
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
                    cv2.imwrite(os.path.join(dst_result_image_dir, f"{imagename}.jpg"), result.visualize())

                    crops, metadata = crop_regions(
                        image=result.image,
                        imagename=imagename,
                        crop_sources=result.anomaly.regions,
                        polygon_sources=result.segmentation.regions,
                        draw_polygon=True,
                    )
                    with open(os.path.join(dst_result_meta_dir, f"{imagename}.json"), "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=4)
                    for filename, (crop, segmentations) in crops.items():
                        cv2.imwrite(os.path.join(dst_crop_images_dir, f"{filename}.jpg"), crop)
                        save_polygons_to_yolo_format(os.path.join(dst_crop_labels_dir, f"{filename}.txt"), [seg["polygon"] for seg in segmentations], [seg["class_id"] for seg in segmentations])

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