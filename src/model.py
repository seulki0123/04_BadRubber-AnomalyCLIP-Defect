import os
import cv2
import tqdm
import time
import torch
import shutil
import numpy as np
from PIL import Image
from ultralytics import YOLO
from types import SimpleNamespace
from scipy.ndimage import gaussian_filter

import draw
import utils as s1k2_utils
from AnomalyCLIP import AnomalyCLIP_lib, utils, prompt_ensemble

class Model:

    def __init__(self, anomalyclip_weights, segmentation_weights, classification_weights, depth=9, n_ctx=12, t_n_ctx=4, image_size=518, features_list=[6, 12, 18, 24], feature_map_layer=[0, 1, 2, 3], sigma=4, DPAM_layer=20):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.depth = depth
        self.n_ctx = n_ctx
        self.t_n_ctx = t_n_ctx
        self.image_size = image_size
        self.features_list = features_list
        self.feature_map_layer = feature_map_layer
        self.sigma = sigma
        self.DPAM_layer = DPAM_layer

        self.anomalyclip_weights = anomalyclip_weights
        self.segmentation_weights = segmentation_weights
        self.classification_weights = classification_weights

        (
            self.anomalyclip_model,
            self.segmentation_model,
            self.classification_model,
            self.text_features,
            self.preprocess,
        ) = self._load()

    def _load(self):

        AnomalyCLIP_parameters = {"Prompt_length": self.n_ctx, "learnabel_text_embedding_depth": self.depth, "learnabel_text_embedding_length": self.t_n_ctx}
        
        anomalyclip_model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=self.device, design_details = AnomalyCLIP_parameters, download_root="./checkpoints/.cache/clip")
        anomalyclip_model.eval()

        preprocess, target_transform = utils.get_transform(SimpleNamespace(image_size=self.image_size))


        prompt_learner = prompt_ensemble.AnomalyCLIP_PromptLearner(anomalyclip_model.to("cpu"), AnomalyCLIP_parameters)
        checkpoint = torch.load(self.anomalyclip_weights)
        prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        prompt_learner.to(self.device)
        anomalyclip_model.to(self.device)
        anomalyclip_model.visual.DAPM_replace(DPAM_layer = self.DPAM_layer)

        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
        text_features = anomalyclip_model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)

        segmentation_model = YOLO(self.segmentation_weights)
        classification_model = YOLO(self.classification_weights)
        print(classification_model.names)
        self.classification_model_names = {0: 'crack', 1: 'pressed', 2: 'trash', 3: 'wet', 4: 'other_rubber', 5: 'trash', 6: 'trash', 7: 'powder', 8: 'trash', 9: 'trash', 10: 'trash', 11: 'wet', 12: 'trash'}
        input(self.classification_model_names)

        return anomalyclip_model, segmentation_model, classification_model, text_features, preprocess

    def warmup(self, iters=10):
        dummy = torch.randn(
            1, 3, self.image_size, self.image_size,
            device=self.device
        )

        with torch.no_grad():
            for _ in tqdm.tqdm(range(iters), desc="Warmup"):
                image_features, patch_features = self.anomalyclip_model.encode_image(
                    dummy,
                    self.features_list,
                    DPAM_layer=self.DPAM_layer
                )

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # text similarity까지 태워서 완전 웜업
                _ = image_features @ self.text_features.permute(0, 2, 1)

        print("Warmup completed")

    def pred(
        self,
        image_path,
        save_dir=None
    ):
        img = Image.open(image_path)

        torch.cuda.synchronize()
        t0 = time.time()

        img_320 = img.copy().resize((320, 320), resample=Image.BILINEAR)
        img = self.preprocess(img)
        
        print("img", img.shape)
        image = img.reshape(1, 3, self.image_size, self.image_size).to(self.device)

        # AnomalyCLIP
        with torch.no_grad():
            image_features, patch_features = self.anomalyclip_model.encode_image(image, self.features_list, DPAM_layer = self.DPAM_layer)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ self.text_features.permute(0, 2, 1)
            text_probs = (text_probs/0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= self.feature_map_layer[0]:
                    patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, self.text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], self.image_size)
                    anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list)
            
            anomaly_map = anomaly_map.sum(dim = 0)
        
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = self.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )

        # YOLOv11 Segmentation
        seg_results = self.segmentation_model(img_320, verbose=False)
        r = seg_results[0]

        if r.masks is None:
            yolo_mask = None
        else:
            mask = r.masks.data[0].cpu().numpy()  # (320, 320), float {0,1}
            mask = cv2.resize(
                mask.astype(np.uint8),
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST
            )
            yolo_mask = mask  # (H, W), {0,1}

        dt = time.time() - t0
        print(f"Prediction time: {dt*1000} ms")

        # Draw
        if save_dir is not None:
            return self.draw(image_path, anomaly_map.detach().cpu().numpy(), save_dir, yolo_mask)

    def draw(self, image_path, anomaly_map, save_dir, yolo_mask):
        filename = os.path.basename(image_path)
        anomaly_map_dir = os.path.join(save_dir, 'anomaly_map')
        segmentation_dir = os.path.join(save_dir, 'segmentation')
        segmentation_meta_dir = os.path.join(save_dir, 'segmentation-meta')
        crops_dir = os.path.join(save_dir, 'crops')
        os.makedirs(anomaly_map_dir, exist_ok=True)
        os.makedirs(segmentation_dir, exist_ok=True)
        os.makedirs(segmentation_meta_dir, exist_ok=True)
        os.makedirs(crops_dir, exist_ok=True)

        # ### original
        image = cv2.resize(cv2.imread(image_path), (self.image_size, self.image_size))
        # shutil.copy(image_path, os.path.join(save_dir, f'{filename}_1_original.jpg'))

        ### anomaly mask
        # vis_anomaly_map = draw.draw_anomaly_map(image, anomaly_map)
        # cv2.imwrite(os.path.join(anomaly_map_dir, f'{filename}_2_anomaly_map.jpg'), vis_anomaly_map)

        ### anomaly mask && yolo mask
        # 1. anomaly mask
        anomaly_mask = draw.get_segmentation_mask(anomaly_map, threshold=0.5)
        anomaly_mask_bin = (anomaly_mask > 0).astype(np.uint8)

        # 2. anomaly ∧ yolo
        if yolo_mask is not None:
            final_mask = anomaly_mask_bin & yolo_mask
            final_mask = anomaly_mask_bin & yolo_mask
        else:
            final_mask = anomaly_mask_bin

        final_mask = final_mask.astype(np.uint8) * 255

        # 3. bbox는 반드시 final_mask에서
        bboxes = draw.get_bboxes_from_mask(final_mask, min_area=300)

        ### crops
        crops = draw.crop_bboxes(image, bboxes, scale=2.0)

        labels = []
        confs = []
        keep_indices = []  # trash 아닌 것만 유지

        for i, crop in enumerate(crops):
            # ratio = s1k2_utils.black_pixel_ratio(crop)
            # cv2.imwrite(os.path.join(crops_dir, f'{ratio:.2f}_{filename}_4_crop_{i}.jpg'), crop)

            results = self.classification_model(crop, verbose=False)
            r = results[0]

            class_number = int(r.probs.top1)
            label = self.classification_model_names[class_number]
            conf = float(r.probs.top1conf)

            if label == "trash":
                continue

            if conf < 0.8:
                continue

            labels.append(label)
            confs.append(conf)
            keep_indices.append(i)

        bboxes = [bboxes[i] for i in keep_indices]

        ### draw masks, bboxes
        # vis_mask_anomaly = draw.draw_segmentation_outline(image, anomaly_mask, color=(0, 0, 255))
        # vis_mask_yolo = draw.draw_segmentation_outline(image, yolo_mask, color=(0, 255, 0)) if yolo_mask is not None else image.copy()
        vis_mask_final = draw.draw_segmentation_outline(image, final_mask, color=(255, 0, 0))
        vis_bbox = draw.draw_bboxes_with_labels(vis_mask_final, bboxes, [f"{labels[i]}: {confs[i]:.2f}" for i in range(len(labels))])
        cv2.imwrite(os.path.join(segmentation_dir, f'{filename}_3_segmentation.jpg'), vis_bbox)

        if len(labels):
            with open(os.path.join(segmentation_meta_dir, f'{filename}_3_segmentation.txt'), 'w') as f:
                f.write(",".join(labels))

        return labels