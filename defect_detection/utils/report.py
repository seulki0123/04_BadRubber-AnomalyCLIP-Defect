class Report:
    def __init__(self):
        self.ok = 0
        self.ng = 0
        self.seg_cnts = {}

        self.error_images = []

    def update(self, labels):
        if len(labels) > 0:
            self.ng += 1
        else:
            self.ok += 1

        for label in labels:
            self.seg_cnts[label] = self.seg_cnts.get(label, 0) + 1
        
    def print_report(self):
        total_rubber = self.ok + self.ng
        total_segs = sum(self.seg_cnts.values())
        print("\n================================================")
        print(f"적합EA: {self.ok}")
        print(f"부적합EA: {self.ng}")
        print(f"전체검출수EA: {total_rubber}")
        print(f"Bale평균검출수EA: {total_segs/ total_rubber}")
        print(f"적합per: {self.ok/ total_rubber *100:.2f} %")
        print(f"부적합per: {self.ng/ total_rubber *100:.2f} %")
        print(f"Class검출수: {self.seg_cnts}")
        print("================================================\n")
