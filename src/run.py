import os

import cv2

from rubber_inspection import Inspector

if __name__ == "__main__":
    # 1. input datas
    img_dir = "./tests/dot"
    imgs_path = [os.path.join(img_dir, img_path) for img_path in os.listdir(img_dir) if img_path.endswith(".jpg")]
    dst_dir = "./tests/dot_results"
    os.makedirs(dst_dir, exist_ok=True)

    # 2. inspect
    inspector = Inspector()
    results = inspector.inspect(imgs_path)

    # 3. visualize
    for result in results:
        image_name = os.path.basename(result.image_path)
        cv2.imwrite(os.path.join(dst_dir, image_name), result.visualize())