import os
import cv2
import tqdm
import time
import torch
import shutil
from PIL import Image
from types import SimpleNamespace
from scipy.ndimage import gaussian_filter

import draw
from AnomalyCLIP import AnomalyCLIP_lib, utils, prompt_ensemble

class Model:

    def __init__(self, checkpoint_path, depth=9, n_ctx=12, t_n_ctx=4, image_size=518, features_list=[6, 12, 18, 24], feature_map_layer=[0, 1, 2, 3], sigma=4, DPAM_layer=20):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.depth = depth
        self.n_ctx = n_ctx
        self.t_n_ctx = t_n_ctx
        self.image_size = image_size
        self.features_list = features_list
        self.feature_map_layer = feature_map_layer
        self.sigma = sigma
        self.DPAM_layer = DPAM_layer

        self.checkpoint_path = checkpoint_path
        (
            self.model,
            self.text_features,
            self.preprocess,

        ) = self._load()

    def _load(self):

        AnomalyCLIP_parameters = {"Prompt_length": self.n_ctx, "learnabel_text_embedding_depth": self.depth, "learnabel_text_embedding_length": self.t_n_ctx}
        
        model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=self.device, design_details = AnomalyCLIP_parameters, download_root="./checkpoints/.cache/clip")
        model.eval()

        preprocess, target_transform = utils.get_transform(SimpleNamespace(image_size=self.image_size))


        prompt_learner = prompt_ensemble.AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
        checkpoint = torch.load(self.checkpoint_path)
        prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        prompt_learner.to(self.device)
        model.to(self.device)
        model.visual.DAPM_replace(DPAM_layer = self.DPAM_layer)

        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
        text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)

        return model, text_features, preprocess

    def warmup(self, iters=10):
        dummy = torch.randn(
            1, 3, self.image_size, self.image_size,
            device=self.device
        )

        with torch.no_grad():
            for _ in tqdm.tqdm(range(iters), desc="Warmup"):
                image_features, patch_features = self.model.encode_image(
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

        img = self.preprocess(img)
        
        print("img", img.shape)
        image = img.reshape(1, 3, self.image_size, self.image_size).to(self.device)


        with torch.no_grad():
            image_features, patch_features = self.model.encode_image(image, self.features_list, DPAM_layer = self.DPAM_layer)
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

        dt = time.time() - t0
        print(f"Prediction time: {dt*1000} ms")

        if save_dir is not None:
            self.draw(image_path, anomaly_map.detach().cpu().numpy(), save_dir)

    def draw(self, image_path, anomaly_map, save_dir):
        filename = os.path.basename(image_path)
        anomaly_map_dir = os.path.join(save_dir, 'anomaly_map')
        segmentation_dir = os.path.join(save_dir, 'segmentation')
        crops_dir = os.path.join(save_dir, 'crops')
        os.makedirs(anomaly_map_dir, exist_ok=True)
        os.makedirs(segmentation_dir, exist_ok=True)
        os.makedirs(crops_dir, exist_ok=True)

        # ### original
        image = cv2.resize(cv2.imread(image_path), (self.image_size, self.image_size))
        # shutil.copy(image_path, os.path.join(save_dir, f'{filename}_1_original.jpg'))

        ### anomaly ma
        vis_anomaly_map = draw.draw_anomaly_map(image, anomaly_map)
        cv2.imwrite(os.path.join(anomaly_map_dir, f'{filename}_2_anomaly_map.jpg'), vis_anomaly_map)

        ### segmentation + bboxes
        mask = draw.get_segmentation_mask(anomaly_map, threshold=0.5)
        bboxes = draw.get_bboxes_from_mask(mask, min_area=300)
        vis_mask = draw.draw_segmentation_outline(image, mask)
        vis_bbox = draw.draw_bboxes(vis_mask, bboxes)
        cv2.imwrite(os.path.join(segmentation_dir, f'{filename}_3_segmentation.jpg'), vis_bbox)

        ### crops
        crops = draw.crop_bboxes(image, bboxes, scale=2.0)
        for i, crop in enumerate(crops):
            ratio = draw.black_pixel_ratio(crop)
            cv2.imwrite(os.path.join(crops_dir, f'{ratio:.2f}_{filename}_4_crop_{i}.jpg'), crop)