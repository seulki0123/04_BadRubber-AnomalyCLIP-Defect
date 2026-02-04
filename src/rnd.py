import random
import os
import shutil
import tqdm

src_dir = "./NAS/dataset_260204_F2150-M2520-F1038/dataset_final_for_anomaly/F2150"
dst_dir = "./NAS/dataset_260204_F2150-M2520-F1038/dataset_final_for_anomaly/F2150_rnd"

rnd_num = 800
os.makedirs(dst_dir, exist_ok=True)
images = os.listdir(src_dir)
random.shuffle(images)
for image in images[:rnd_num]:
    shutil.copy(os.path.join(src_dir, image), os.path.join(dst_dir, image))