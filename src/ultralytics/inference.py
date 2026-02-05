from ultralytics import YOLO
import os
import shutil

# 모델 로드
model_path = "/home/ssr/1_LG/02_code/yolov8n-seg/weights/best.pt"
model = YOLO(model_path)

# 테스트 이미지 폴더
test_image_folder = "/home/ssr/1_LG/NAS_S1K2/BadRuuber/dataset_260204_F2150-M2520-F1038/dataset_notnull/F2150/images"

# 결과 저장 폴더
output_root = "/home/ssr/1_LG/NAS_S1K2/BadRuuber/dataset_260204_F2150-M2520-F1038/dataset_notnull_bgseg/F2150"
output_name = "results"
output_folder = os.path.join(output_root, output_name)
os.makedirs(output_folder, exist_ok=True)

# 예측 (이미지 저장 X, txt만 저장)
results = model.predict(
    source=test_image_folder,
    save=True,        # 이미지 바로 저장 X
    save_txt=True,     # txt 저장
    save_conf=True,
    project=output_root,
    name=output_name,
    exist_ok=True,
    batch=8,
    stream=False
)

# 탐지된 이미지만 원본 저장
images_output_folder = os.path.join(output_folder, "images_with_detection")
os.makedirs(images_output_folder, exist_ok=True)

for result in results:
    # result.masks 존재 = 탐지 있음
    if result.masks is not None and len(result.masks.data) > 0:
        # 원본 이미지 복사
        src_path = result.path
        dst_path = os.path.join(images_output_folder, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)

print("탐지된 이미지 + txt 파일 저장 완료!")