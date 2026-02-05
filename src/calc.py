import os
import tqdm

if __name__ == "__main__":

    date = "2026-01-04"
    kind = None
    src_dir = f"./NAS_Site_SSBR/{date}"
    dst_dir = f"./NAS/classify/datasets/raw/M2520/{date}"

    TotalEA = len([i for i in os.listdir(src_dir) if i.endswith(".jpg") and kind in i if kind is not None else True])

    적합EA = TotalEA
    부적합EA = 0

    전체검출수EA = 0
    Bale평균검출수EA = 0

    적합per = 0
    부적합per = 0

    Class검출수 = {}

    def pinrt2():
        print(f"TotalEA: {TotalEA}")
        print(f"적합EA: {적합EA}")
        print(f"부적합EA: {부적합EA}")
        print(f"전체검출수EA: {전체검출수EA}")
        print(f"Bale평균검출수EA: {Bale평균검출수EA}")
        print(f"적합per: {적합per:.4f}")
        print(f"부적합per: {부적합per:.4f}")
        print(f"Class검출수: {Class검출수}")
        print("-"*100, "\n")

    idx = 0
    for text_file in tqdm.tqdm(os.listdir(dst_dir)):
        if not text_file.endswith(".txt"):
            continue

        with open(os.path.join(dst_dir, text_file), 'r') as f:
            labels = f.read().split(',')

        if len(labels):
            부적합EA += 1
            적합EA = TotalEA - 부적합EA

        전체검출수EA += len(labels)
        Bale평균검출수EA = 전체검출수EA / TotalEA

        적합per = 적합EA / TotalEA
        부적합per = 부적합EA / TotalEA

        for label in labels:
            Class검출수[label] = Class검출수.get(label, 0) + 1

        # if idx % 100 == 0:
        if True:
            pinrt2()

        idx += 1
    
    print2()