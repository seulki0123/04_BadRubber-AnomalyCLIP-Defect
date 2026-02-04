import os
import shutil
import tqdm

src_dir = "./datasets/F2150_1classTest-1/test"
dst_dir = "./datasets/F2150_1classTest-1_nullX"
os.makedirs(dst_dir, exist_ok=True)

images = os.listdir(os.path.join(src_dir, "images"))
labels = os.listdir(os.path.join(src_dir, "labels"))
nulls_cnt = 0
not_nulls_cnt = 0
for label in tqdm.tqdm(labels):
    if not label.endswith(".txt"):
        continue

    with open(os.path.join(src_dir, "labels", label), "r") as f:
        lines = f.read()

    if lines:
        not_nulls_cnt += 1
        shutil.copy(os.path.join(src_dir, "images", label.replace(".txt", ".jpg")), os.path.join(dst_dir))
    else:
        nulls_cnt += 1

print(f"Nulls: {nulls_cnt}, Not nulls: {not_nulls_cnt}")