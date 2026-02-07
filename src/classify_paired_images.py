# -*- coding: utf-8 -*-
"""위/아래 사진 쌍으로 생성공정 분류 (즉시 분류 가능)"""
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
import json
import re
import shutil
from datetime import datetime
from typing import Tuple, Optional

from config import get_config


def parse_filename(filename: str) -> Optional[Tuple[str, str, datetime]]:
    """파일명에서 타입, 카메라 번호, 타임스탬프 추출
    예: BR-C_1_20251112_131532_974.jpg -> (BR, 1, datetime)
    예: BR-A_1_20251113_141118_540.jpg -> (BR-A, 1, datetime)
    예: BR-B_1_20251113_141118_540.jpg -> (BR-B, 1, datetime)
    예: SSBR_1_20251113_104159_232.jpg -> (SSBR, 1, datetime)
    예: NBR_1_20251125_002908_295.jpg -> (NBR, 1, datetime)
    """
    # 패턴 1: BR-C 형식: {타입}-C_{카메라번호}_{날짜}_{시간}_{밀리초}.jpg
    pattern1 = r'^([A-Z]+)-C_(\d+)_(\d{8})_(\d{6})_(\d+)\.jpg$'
    match = re.match(pattern1, filename)
    
    if match:
        rubber_type = match.group(1)
        camera_num = match.group(2)
        date_str = match.group(3)
        time_str = match.group(4)
        millisec = match.group(5)
    else:
        # 패턴 2: BR-A 형식: {타입}-A_{카메라번호}_{날짜}_{시간}_{밀리초}.jpg
        pattern2 = r'^([A-Z]+)-A_(\d+)_(\d{8})_(\d{6})_(\d+)\.jpg$'
        match = re.match(pattern2, filename)
        if match:
            rubber_type = f"{match.group(1)}-A"  # BR-A 형식 유지
            camera_num = match.group(2)
            date_str = match.group(3)
            time_str = match.group(4)
            millisec = match.group(5)
        else:
            # 패턴 3: BR-B 형식: {타입}-B_{카메라번호}_{날짜}_{시간}_{밀리초}.jpg
            pattern3 = r'^([A-Z]+)-B_(\d+)_(\d{8})_(\d{6})_(\d+)\.jpg$'
            match = re.match(pattern3, filename)
            if match:
                rubber_type = f"{match.group(1)}-B"  # BR-B 형식 유지
                camera_num = match.group(2)
                date_str = match.group(3)
                time_str = match.group(4)
                millisec = match.group(5)
            else:
                # 패턴 4: SSBR 형식: SSBR_{카메라번호}_{날짜}_{시간}_{밀리초}.jpg
                pattern4 = r'^SSBR_(\d+)_(\d{8})_(\d{6})_(\d+)\.jpg$'
                match = re.match(pattern4, filename)
                if match:
                    rubber_type = "SSBR"
                    camera_num = match.group(1)
                    date_str = match.group(2)
                    time_str = match.group(3)
                    millisec = match.group(4)
                else:
                    # 패턴 5: NBR 형식: NBR_{카메라번호}_{날짜}_{시간}_{밀리초}.jpg
                    pattern5 = r'^NBR_(\d+)_(\d{8})_(\d{6})_(\d+)\.jpg$'
                    match = re.match(pattern5, filename)
                    if not match:
                        return None
                    
                    rubber_type = "NBR"
                    camera_num = match.group(1)
                    date_str = match.group(2)
                    time_str = match.group(3)
                    millisec = match.group(4)
    
    try:
        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        try:
            millisec_int = int(millisec)
            dt = dt.replace(microsecond=millisec_int * 1000)
        except ValueError:
            dt = dt.replace(microsecond=0)
        return (rubber_type, camera_num, dt)
    except (ValueError, OverflowError):
        return None


def load_image_pairs_from_folder(input_dir: str, max_time_gap_seconds: float = 5.0):
    """폴더에서 이미지 파일들을 읽어서 윗부분/아랫부분 쌍으로 묶기
    
    Args:
        input_dir: 입력 디렉토리 경로 (예: temp_images/BR-C1)
        max_time_gap_seconds: 쌍으로 묶을 수 있는 최대 시간 간격 (초)
    
    Returns:
        [(윗부분_파일경로, 아랫부분_파일경로), ...] 리스트
    """
    if not os.path.exists(input_dir):
        return []
    
    # 이미지 파일 목록
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
    
    if len(files) == 0:
        return []
    
    # 파일명 파싱 및 정렬
    parsed_files = []
    for f in files:
        parsed = parse_filename(f)
        if parsed:
            rubber_type, camera_num, dt = parsed
            parsed_files.append((f, rubber_type, camera_num, dt))
    
    if len(parsed_files) == 0:
        return []
    
    # 타입과 카메라별로 그룹화
    groups = {}
    for f, rubber_type, camera_num, dt in parsed_files:
        key = (rubber_type, camera_num)
        if key not in groups:
            groups[key] = []
        groups[key].append((f, dt))
    
    # 각 그룹 내에서 타임스탬프로 정렬
    for key in groups:
        groups[key].sort(key=lambda x: x[1])
    
    # 쌍 만들기
    pairs = []
    for (rubber_type, camera_num), file_list in groups.items():
        i = 0
        while i < len(file_list) - 1:
            f1, dt1 = file_list[i]
            f2, dt2 = file_list[i + 1]
            
            # 시간 간격 계산 (초)
            time_gap = abs((dt2 - dt1).total_seconds())
            
            # 시간 간격이 허용 범위 내이면 쌍으로 묶기
            if 0 < time_gap <= max_time_gap_seconds:
                # 정렬된 순서: f1(인덱스 i)가 먼저, f2(인덱스 i+1)가 나중
                # 하지만 실제로는 나중에 찍힌 것이 윗부분이므로 순서 반대
                # f2가 윗부분(나중에 찍힌 것), f1이 아랫부분(먼저 찍힌 것)
                top_path = os.path.join(input_dir, f2)
                bottom_path = os.path.join(input_dir, f1)
                pairs.append((top_path, bottom_path))
                i += 2  # 두 파일 모두 사용했으므로 다음으로
            else:
                # 시간 간격이 너무 크면 첫 번째 파일은 건너뛰기
                i += 1
    
    return pairs


def generate_pair_id(top_img_path: str, bottom_img_path: str) -> str:
    """위/아래 이미지 경로에서 쌍 ID 생성
    
    Args:
        top_img_path: 위 이미지 경로
        bottom_img_path: 아래 이미지 경로
    
    Returns:
        쌍 ID (예: BR-C_1_20251112_131532)
    """
    top_filename = os.path.basename(top_img_path)
    bottom_filename = os.path.basename(bottom_img_path)
    
    # 위 이미지에서 타임스탬프 추출
    top_parsed = parse_filename(top_filename)
    bottom_parsed = parse_filename(bottom_filename)
    
    if top_parsed and bottom_parsed:
        # 두 타임스탬프 중 더 나중 것을 사용 (윗부분)
        rubber_type, camera_num, top_dt = top_parsed
        _, _, bottom_dt = bottom_parsed
        
        # 더 나중 타임스탬프 사용
        if top_dt >= bottom_dt:
            dt = top_dt
        else:
            dt = bottom_dt
        
        # 타임스탬프를 문자열로 변환
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        # 타입별 형식 처리
        if rubber_type == "SSBR":
            pair_id = f"SSBR_{camera_num}_{timestamp_str}"
        elif rubber_type == "NBR":
            pair_id = f"NBR_{camera_num}_{timestamp_str}"
        elif "-A" in rubber_type or "-B" in rubber_type:
            # BR-A, BR-B 형식
            pair_id = f"{rubber_type}_{camera_num}_{timestamp_str}"
        else:
            # BR-C 등 기본 형식
            pair_id = f"{rubber_type}-C_{camera_num}_{timestamp_str}"
        return pair_id
    else:
        # 파싱 실패 시 파일명 기반으로 생성
        # 두 파일명의 공통 부분 추출
        top_name = os.path.splitext(top_filename)[0]
        bottom_name = os.path.splitext(bottom_filename)[0]
        
        # 공통 부분 찾기
        common_parts = []
        top_parts = top_name.split('_')
        bottom_parts = bottom_name.split('_')
        
        # 앞부분 공통 요소 찾기
        for i in range(min(len(top_parts), len(bottom_parts))):
            if top_parts[i] == bottom_parts[i]:
                common_parts.append(top_parts[i])
            else:
                break
        
        if common_parts:
            pair_id = '_'.join(common_parts)
        else:
            # 공통 부분이 없으면 타임스탬프 사용
            pair_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        return pair_id


def build_model(num_classes=3, backbone='resnet18'):
    """모델 생성"""
    if backbone == 'resnet18':
        import torchvision.models as models
        model = models.resnet18(weights=None)  # 학습된 가중치를 로드할 것이므로 None
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    elif backbone == 'efficientnet':
        try:
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained('efficientnet-b0')
            num_features = model._fc.in_features
            model._fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        except ImportError:
            print("EfficientNet이 설치되지 않았습니다. ResNet18을 사용합니다.")
            import torchvision.models as models
            model = models.resnet18(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
    return model


class ProcessClassifier:
    """생성공정 분류기 (위/아래 사진 쌍)"""
    def __init__(self, model_dir: str, device: str = 'auto', backbone: str = 'resnet18', img_size: int = 224, num_classes: int = None):
        """
        Args:
            model_dir: 모델 디렉토리 경로 (runs_process_classifier_temp_images)
            device: 디바이스 ('auto', 'cuda', 'cpu')
            backbone: 백본 모델 ('resnet18', 'efficientnet')
            img_size: 이미지 크기
            num_classes: 클래스 수 (None이면 체크포인트에서 자동 감지)
        """
        self.model_dir = model_dir
        self.img_size = img_size
        self.backbone = backbone
        self.num_classes = num_classes  # None이면 체크포인트에서 자동 감지
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"디바이스: {self.device}")
        
        # 이미지 전처리 (이미지가 이미 img_size x img_size이므로 resize 제거)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 모델 로드
        self.models = []
        self.load_models()
        
        # 공정 레이블 (모델 경로와 num_classes에 따라 동적으로 설정)
        # 모델 경로에서 production type 감지
        model_dir_lower = self.model_dir.lower()
        if 'br_c' in model_dir_lower or 'br-c' in model_dir_lower:
            # BR-C 모델: 3개 클래스면 {0: '0', 1: '1', 2: 'A'}
            if self.num_classes == 3:
                self.process_labels = {0: '0', 1: '1', 2: 'A'}
            elif self.num_classes == 2:
                self.process_labels = {0: '0', 1: 'A'}  # 2개 클래스인 경우
            else:
                self.process_labels = {i: str(i) for i in range(self.num_classes)}
        elif 'br_a' in model_dir_lower or 'br-a' in model_dir_lower:
            # BR-A 모델: 2개 클래스면 {0: '0', 1: '2'}
            if self.num_classes == 2:
                self.process_labels = {0: '0', 1: '2'}
            else:
                self.process_labels = {i: str(i) for i in range(self.num_classes)}
        elif 'br_b' in model_dir_lower or 'br-b' in model_dir_lower:
            # BR-B 모델: 2개 클래스면 {0: '1', 1: '3'}
            if self.num_classes == 2:
                self.process_labels = {0: '1', 1: '3'}
            else:
                self.process_labels = {i: str(i) for i in range(self.num_classes)}
        elif 'nbr' in model_dir_lower:
            # NBR 모델: 2개 클래스면 {0: '0', 1: '1'}
            if self.num_classes == 2:
                self.process_labels = {0: '0', 1: '1'}
            else:
                self.process_labels = {i: str(i) for i in range(self.num_classes)}
        elif self.num_classes == 4:
            # SSBR 4클래스 모델
            self.process_labels = {0: '0', 1: '1', 2: '2', 3: 'A'}
        else:
            # 기본값: SSBR 3클래스 모델 {0: '0', 1: '1', 2: '2'}
            self.process_labels = {0: '0', 1: '1', 2: '2'}
    
    def load_models(self):
        """K-Fold 모델들 또는 단일 모델 로드"""
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"모델 디렉토리가 없습니다: {self.model_dir}")
        
        # 단일 모델 확인 (model_dir에 직접 best_model.pth가 있는 경우)
        single_model_path = os.path.join(self.model_dir, 'best_model.pth')
        if os.path.exists(single_model_path):
            print(f"모델 로드 시작 (디바이스={self.device})")
            print("="*60)
            print("단일 모델 로드 중...")
            print("="*60)
            
            # 체크포인트 로드 (한 번만 읽기)
            try:
                checkpoint = torch.load(single_model_path, map_location=self.device, weights_only=True)
            except Exception as e:
                # PyTorch 2.6+ weights_only=True 실패 시 False로 재시도
                checkpoint = torch.load(single_model_path, map_location=self.device, weights_only=False)
            
            # num_classes 자동 감지
            if self.num_classes is None:
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    # 마지막 FC 레이어 찾기 (가장 큰 인덱스를 가진 fc 레이어)
                    fc_weight_keys = [k for k in state_dict.keys() if 'fc' in k and 'weight' in k]
                    if fc_weight_keys:
                        # fc.0.weight, fc.1.weight, fc.2.weight 등 중에서
                        # 숫자가 가장 큰 것이 마지막 레이어
                        def get_fc_index(key):
                            # fc.1.weight -> 1, fc.4.weight -> 4
                            import re
                            match = re.search(r'fc\.(\d+)\.weight', key)
                            return int(match.group(1)) if match else -1
                        
                        # 인덱스로 정렬하여 마지막 레이어 찾기
                        fc_weight_keys_sorted = sorted(fc_weight_keys, key=get_fc_index)
                        last_fc_key = fc_weight_keys_sorted[-1]
                        self.num_classes = state_dict[last_fc_key].shape[0]
                        print(f"[정보] 마지막 FC 레이어 감지: {last_fc_key}, 클래스 수: {self.num_classes}")
                    else:
                        # fc 레이어를 찾지 못한 경우 기본값 사용
                        self.num_classes = 3
                        print(f"[경고] FC 레이어를 찾지 못해 기본값 사용: {self.num_classes}")
                else:
                    self.num_classes = 3  # 기본값
                    print(f"[경고] model_state_dict를 찾지 못해 기본값 사용: {self.num_classes}")
                print(f"[정보] 체크포인트에서 자동 감지된 클래스 수: {self.num_classes}")
            
            # 모델 구조 자동 감지: fc.weight가 있으면 단일 Linear, fc.4.weight가 있으면 Sequential
            state_dict_keys = checkpoint['model_state_dict'].keys()
            has_simple_fc = 'fc.weight' in state_dict_keys and 'fc.bias' in state_dict_keys
            has_sequential_fc = 'fc.1.weight' in state_dict_keys or 'fc.4.weight' in state_dict_keys
            
            # 모델 생성 (구조에 맞게)
            if has_simple_fc and not has_sequential_fc:
                # 단일 Linear 레이어 구조 (F1810 학습 모델과 동일)
                import torchvision.models as models
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            else:
                # Sequential 구조 (기존 구조)
                model = build_model(num_classes=self.num_classes, backbone=self.backbone)
            
            model = model.to(self.device)
            
            # 가중치 로드 (이미 읽은 checkpoint 재사용)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.models.append(model)
            print(f"[OK] 단일 모델 로드 완료: {single_model_path}")
            print(f"\n[OK] 총 {len(self.models)}개 모델 로드 완료")
            print("="*60)
            print("모델 로드 완료!")
            print("="*60)
            return
        
        # K-Fold 모델 찾기
        fold_dirs = [d for d in os.listdir(self.model_dir) 
                     if os.path.isdir(os.path.join(self.model_dir, d)) and d.startswith('fold_')]
        fold_dirs.sort()
        
        if len(fold_dirs) == 0:
            raise FileNotFoundError(f"모델을 찾을 수 없습니다: {self.model_dir} (단일 모델 또는 K-Fold 모델이 필요합니다)")
        
        print(f"모델 로드 시작 (디바이스={self.device})")
        print("="*60)
        print("1단계: K-Fold 모델 로드 중...")
        print("="*60)
        
        # 첫 번째 모델에서 num_classes 자동 감지 및 모델 로드
        first_model_path = os.path.join(self.model_dir, fold_dirs[0], 'best_model.pth')
        first_checkpoint = None
        
        if os.path.exists(first_model_path):
            # 첫 번째 체크포인트 로드 (한 번만 읽기)
            try:
                first_checkpoint = torch.load(first_model_path, map_location=self.device, weights_only=True)
            except Exception as e:
                # PyTorch 2.6+ weights_only=True 실패 시 False로 재시도
                first_checkpoint = torch.load(first_model_path, map_location=self.device, weights_only=False)
            
            # num_classes 자동 감지
            if self.num_classes is None:
                if 'model_state_dict' in first_checkpoint:
                    state_dict = first_checkpoint['model_state_dict']
                    # 마지막 FC 레이어 찾기 (가장 큰 인덱스를 가진 fc 레이어)
                    fc_weight_keys = [k for k in state_dict.keys() if 'fc' in k and 'weight' in k]
                    if fc_weight_keys:
                        # fc.0.weight, fc.1.weight, fc.2.weight 등 중에서
                        # 숫자가 가장 큰 것이 마지막 레이어
                        def get_fc_index(key):
                            # fc.1.weight -> 1, fc.4.weight -> 4
                            import re
                            match = re.search(r'fc\.(\d+)\.weight', key)
                            return int(match.group(1)) if match else -1
                        
                        # 인덱스로 정렬하여 마지막 레이어 찾기
                        fc_weight_keys_sorted = sorted(fc_weight_keys, key=get_fc_index)
                        last_fc_key = fc_weight_keys_sorted[-1]
                        self.num_classes = state_dict[last_fc_key].shape[0]
                        print(f"[정보] 마지막 FC 레이어 감지: {last_fc_key}, 클래스 수: {self.num_classes}")
                    else:
                        # fc 레이어를 찾지 못한 경우 기본값 사용
                        self.num_classes = 3
                        print(f"[경고] FC 레이어를 찾지 못해 기본값 사용: {self.num_classes}")
                else:
                    self.num_classes = 3  # 기본값
                    print(f"[경고] model_state_dict를 찾지 못해 기본값 사용: {self.num_classes}")
                print(f"[정보] 체크포인트에서 자동 감지된 클래스 수: {self.num_classes}")
        
        for idx, fold_dir in enumerate(fold_dirs):
            model_path = os.path.join(self.model_dir, fold_dir, 'best_model.pth')
            if not os.path.exists(model_path):
                print(f"[경고] 모델 파일이 없습니다: {model_path}")
                continue
            
            # 첫 번째 모델은 이미 읽은 checkpoint 재사용, 나머지는 새로 읽기
            if idx == 0 and first_checkpoint is not None:
                checkpoint = first_checkpoint
            else:
                try:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                except Exception as e:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 모델 구조 자동 감지
            state_dict_keys = checkpoint['model_state_dict'].keys()
            has_simple_fc = 'fc.weight' in state_dict_keys and 'fc.bias' in state_dict_keys
            has_sequential_fc = 'fc.1.weight' in state_dict_keys or 'fc.4.weight' in state_dict_keys
            
            # 모델 생성 (구조에 맞게)
            if has_simple_fc and not has_sequential_fc:
                # 단일 Linear 레이어 구조
                import torchvision.models as models
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            else:
                # Sequential 구조
                model = build_model(num_classes=self.num_classes, backbone=self.backbone)
            
            model = model.to(self.device)
            
            # 가중치 로드
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.models.append(model)
            print(f"[OK] {fold_dir} 모델 로드 완료: {model_path}")
        
        if len(self.models) == 0:
            raise FileNotFoundError("로드된 모델이 없습니다.")
        
        print(f"\n[OK] 총 {len(self.models)}개 모델 로드 완료")
        print("="*60)
        print("모델 로드 완료!")
        print("="*60)
    
    def combine_images(self, top_img: Image.Image, bottom_img: Image.Image) -> Image.Image:
        """위, 아래 이미지를 결합
        - top 이미지: 아랫쪽 절반 크롭 (고무 표시가 있는 부분)
        - bottom 이미지: 윗쪽 절반 크롭 (고무 표시가 있는 부분)
        
        Args:
            top_img: 위 이미지
            bottom_img: 아래 이미지
        
        Returns:
            결합된 이미지
        """
        # 원본 이미지 크기
        orig_w, orig_h = top_img.size
        
        # top 이미지: 아랫쪽 절반 크롭 (고무 표시가 있는 부분)
        # 사진의 아랫쪽 부분에 표시가 있으므로 아랫쪽 절반을 가져옴
        top_crop = top_img.crop((0, orig_h // 2, orig_w, orig_h))  # 아랫쪽 절반
        top_crop = top_crop.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        # bottom 이미지: 윗쪽 절반 크롭 (고무 표시가 있는 부분)
        # 사진의 윗쪽 부분에 표시가 있으므로 윗쪽 절반을 가져옴
        bottom_crop = bottom_img.crop((0, 0, orig_w, orig_h // 2))  # 윗쪽 절반
        bottom_crop = bottom_crop.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        # 두 이미지를 위아래로 결합 (학습 시와 동일한 224x448 형태 유지)
        combined_img = Image.new('RGB', (self.img_size, self.img_size * 2))
        combined_img.paste(top_crop, (0, 0))
        combined_img.paste(bottom_crop, (0, self.img_size))
        
        # 학습 시와 동일한 형태로 유지 (resize 제거)
        # 학습 코드: combined = Image.new('RGB', (self.size, self.size * 2))
        # 평가 코드: combined = Image.new('RGB', (self.size, self.size * 2))
        
        return combined_img
    
    def classify(self, top_img_path: str, bottom_img_path: str) -> dict:
        """위/아래 사진 쌍으로 생성공정 분류
        
        Args:
            top_img_path: 위 이미지 경로 (또는 첫 번째 이미지)
            bottom_img_path: 아래 이미지 경로 (또는 두 번째 이미지)
        
        Returns:
            {
                'process': int,  # 공정 번호 (0, 1, 2)
                'process_label': str,  # 공정 레이블 ('0', '1', 'A')
                'probabilities': dict,  # 각 공정별 확률
                'confidence': float,  # 예측 신뢰도
                'top_img_path': str,  # 위 이미지 경로 (타임스탬프가 더 나중)
                'bottom_img_path': str,  # 아래 이미지 경로 (타임스탬프가 더 먼저)
                'pair_id': str  # 쌍 ID
            }
        """
        # 파일 존재 확인
        if not os.path.exists(top_img_path):
            raise FileNotFoundError(f"위 이미지를 찾을 수 없습니다: {top_img_path}")
        if not os.path.exists(bottom_img_path):
            raise FileNotFoundError(f"아래 이미지를 찾을 수 없습니다: {bottom_img_path}")
        
        # 타임스탬프 기반으로 위/아래 순서 자동 보정
        # 학습 데이터와 동일하게: 나중에 찍힌 것이 위(top), 먼저 찍힌 것이 아래(bottom)
        original_top_path = top_img_path
        original_bottom_path = bottom_img_path
        
        top_filename = os.path.basename(top_img_path)
        bottom_filename = os.path.basename(bottom_img_path)
        
        top_parsed = parse_filename(top_filename)
        bottom_parsed = parse_filename(bottom_filename)
        
        # 타임스탬프가 있으면 순서 보정
        order_corrected = False
        if top_parsed and bottom_parsed:
            _, _, top_dt = top_parsed
            _, _, bottom_dt = bottom_parsed
            
            # 나중에 찍힌 것이 위(top)에 오도록 보정
            if top_dt < bottom_dt:
                # 순서가 반대이므로 교환
                top_img_path, bottom_img_path = bottom_img_path, top_img_path
                top_filename, bottom_filename = bottom_filename, top_filename
                order_corrected = True
        
        # 이미지 로드
        top_img = Image.open(top_img_path).convert('RGB')
        bottom_img = Image.open(bottom_img_path).convert('RGB')
        
        # 이미지 결합
        combined_img = self.combine_images(top_img, bottom_img)
        
        # 전처리
        img_tensor = self.transform(combined_img).unsqueeze(0).to(self.device)
        
        # 모든 모델로 예측
        all_probs = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        # 평균 확률 계산 (앙상블)
        avg_probs = np.mean(all_probs, axis=0)[0]
        
        # 예측
        predicted_class = int(np.argmax(avg_probs))
        confidence = float(avg_probs[predicted_class])
        
        # 디버깅: 확률 분포 출력 (항상 출력)
        print(f"\n[ProcessClassifier 디버깅] 분류 결과:")
        print(f"  num_classes: {self.num_classes}")
        print(f"  process_labels: {self.process_labels}")
        print(f"  predicted_class (인덱스): {predicted_class}")
        print(f"  confidence: {confidence:.3f}")
        # num_classes에 따라 동적으로 확률 분포 출력
        prob_str = ", ".join([f"클래스{i}({self.process_labels.get(i, str(i))}): {avg_probs[i]:.4f}" for i in range(len(avg_probs))])
        print(f"  확률 분포: [{prob_str}]")
        print(f"  top_img: {os.path.basename(top_img_path)}")
        print(f"  bottom_img: {os.path.basename(bottom_img_path)}")
        
        # 쌍 ID 생성
        pair_id = generate_pair_id(top_img_path, bottom_img_path)
        
        # probabilities 딕셔너리 생성 (num_classes에 따라 동적)
        probabilities = {}
        for i in range(self.num_classes):
            label = self.process_labels.get(i, str(i))
            probabilities[label] = float(avg_probs[i])
        
        # 결과 반환
        result = {
            'process': predicted_class,
            'process_label': self.process_labels[predicted_class],
            'probabilities': probabilities,
            'confidence': confidence,
            'top_img_path': top_img_path,
            'bottom_img_path': bottom_img_path,
            'original_top_path': original_top_path,
            'original_bottom_path': original_bottom_path,
            'order_corrected': order_corrected,
            'pair_id': pair_id
        }
        
        return result
    
    def classify_and_save(self, top_img_path: str, bottom_img_path: str, output_dir: str, save_combined: bool = True) -> dict:
        """위/아래 사진 쌍으로 생성공정 분류하고 결과 저장
        
        Args:
            top_img_path: 위 이미지 경로 (또는 첫 번째 이미지, 타임스탬프 기반 자동 보정)
            bottom_img_path: 아래 이미지 경로 (또는 두 번째 이미지, 타임스탬프 기반 자동 보정)
            output_dir: 결과 저장 디렉토리
            save_combined: 결합 이미지도 저장할지 여부
        
        Returns:
            분류 결과 딕셔너리
        """
        # 분류 수행 (내부에서 타임스탬프 기반 순서 보정됨)
        result = self.classify(top_img_path, bottom_img_path)
        
        # 보정된 경로 사용
        top_img_path = result['top_img_path']
        bottom_img_path = result['bottom_img_path']
        
        # 공정별 폴더 생성
        process_label = result['process_label']
        process_dir = os.path.join(output_dir, process_label)
        os.makedirs(process_dir, exist_ok=True)
        
        # 쌍 ID 추출
        pair_id = result['pair_id']
        
        # 파일명 생성 (쌍 ID + 위/아래 구분)
        top_filename = f"{pair_id}_top.jpg"
        bottom_filename = f"{pair_id}_bottom.jpg"
        combined_filename = f"{pair_id}_combined.jpg"
        
        # 이미지 저장
        top_output = os.path.join(process_dir, top_filename)
        bottom_output = os.path.join(process_dir, bottom_filename)
        
        # 원본 이미지 복사
        shutil.copy2(top_img_path, top_output)
        shutil.copy2(bottom_img_path, bottom_output)
        
        saved_files = {
            'top': top_output,
            'bottom': bottom_output,
            'metadata': None
        }
        
        # 결합 이미지 저장 (선택사항)
        if save_combined:
            top_img = Image.open(top_img_path).convert('RGB')
            bottom_img = Image.open(bottom_img_path).convert('RGB')
            combined_img = self.combine_images(top_img, bottom_img)
            combined_output = os.path.join(process_dir, combined_filename)
            combined_img.save(combined_output, quality=95)
            saved_files['combined'] = combined_output
        
        # 메타데이터 JSON 저장
        metadata = {
            'pair_id': pair_id,
            'process': result['process'],
            'process_label': result['process_label'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'top_img': top_filename,
            'bottom_img': bottom_filename,
            'top_img_path': top_img_path,
            'bottom_img_path': bottom_img_path,
            'original_top_path': result.get('original_top_path', top_img_path),
            'original_bottom_path': result.get('original_bottom_path', bottom_img_path),
            'order_corrected': result.get('order_corrected', False)
        }
        
        if save_combined:
            metadata['combined_img'] = combined_filename
        
        metadata_path = os.path.join(process_dir, f"{pair_id}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        saved_files['metadata'] = metadata_path
        result['saved_files'] = saved_files
        
        return result
    
    def classify_from_paths(self, top_path: str, bottom_path: str) -> dict:
        """경로에서 이미지를 로드하여 분류 (편의 메서드)"""
        return self.classify(top_path, bottom_path)


def main():
    """메인 함수"""
    # config에서 설정 가져오기
    config = get_config()
    
    # 고정된 설정
    DEFAULT_OUTPUT_DIR = 'result_paired'
    DEFAULT_INPUT_DIR = 'temp_images/BR-C1'  # 기본 입력 디렉토리
    DEFAULT_DEVICE = 'auto'
    # config에서 기본값 가져오기
    DEFAULT_BACKBONE = getattr(config, 'classification', None) and config.classification.backbone or 'resnet18'
    DEFAULT_IMG_SIZE = getattr(config, 'classification', None) and config.classification.img_size or 224
    # 기본 모델 경로는 config에서 가져오기 (config.model_paths.model_path 사용)
    DEFAULT_MODEL_DIR = config.model_paths.model_path if hasattr(config, 'model_paths') else None
    if DEFAULT_MODEL_DIR is None or not os.path.exists(DEFAULT_MODEL_DIR):
        # config에 없거나 경로가 존재하지 않으면 기본값 사용
        DEFAULT_MODEL_DIR = 'runs_process_classifier_temp_images/BR-C1'
    
    parser = argparse.ArgumentParser(description='위/아래 사진 쌍으로 생성공정 분류')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR,
                       help=f'입력 디렉토리 경로 (기본: {DEFAULT_INPUT_DIR}). 지정하면 폴더 내 모든 이미지 쌍을 자동 분류')
    parser.add_argument('--top_img', type=str, default=None, 
                       help='위 이미지 경로 (또는 첫 번째 이미지, 타임스탬프 기반 자동 보정). --input_dir와 함께 사용 불가')
    parser.add_argument('--bottom_img', type=str, default=None, 
                       help='아래 이미지 경로 (또는 두 번째 이미지, 타임스탬프 기반 자동 보정). --input_dir와 함께 사용 불가')
    parser.add_argument('--max_time_gap', type=float, default=5.0,
                       help='쌍으로 묶을 수 있는 최대 시간 간격 (초, 기본: 5.0)')
    parser.add_argument('--production', type=str, default=None,
                       choices=['SSBR', 'BR-A', 'BR-C', 'BR-B', 'NBR'],
                       help='생성공정 타입 (지정하면 config에서 해당 모델 경로 자동 사용)')
    parser.add_argument('--model_dir', type=str, default=None,
                       help=f'모델 디렉토리 경로 (기본: config에서 가져오거나 {DEFAULT_MODEL_DIR})')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, choices=['auto', 'cuda', 'cpu'],
                       help=f'디바이스 (기본: {DEFAULT_DEVICE})')
    parser.add_argument('--backbone', type=str, default=DEFAULT_BACKBONE, choices=['resnet18', 'efficientnet'],
                       help=f'백본 모델 (기본: {DEFAULT_BACKBONE})')
    parser.add_argument('--img_size', type=int, default=DEFAULT_IMG_SIZE, help=f'이미지 크기 (기본: {DEFAULT_IMG_SIZE})')
    parser.add_argument('--output', type=str, default=None, help='결과를 JSON 파일로 저장 (선택)')
    parser.add_argument('--save_results', type=str, default=DEFAULT_OUTPUT_DIR, 
                       help=f'분류 결과를 공정별 폴더에 저장 (기본: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--no_save', action='store_true', 
                       help='결과를 저장하지 않고 분류만 수행')
    parser.add_argument('--no_save_combined', action='store_false', dest='save_combined',
                       default=True, help='결합 이미지 저장 안 함 (기본: 저장함)')
    
    args = parser.parse_args()
    
    # 모델 경로 결정: --model_dir이 지정되면 우선, 그 다음 --production, 마지막으로 기본값
    if args.model_dir:
        model_dir = args.model_dir
    elif args.production:
        model_dir = config.get_model_path(args.production)
        if model_dir is None:
            print(f"경고: {args.production}에 대한 모델 경로를 config에서 찾을 수 없습니다. 기본값을 사용합니다.")
            model_dir = DEFAULT_MODEL_DIR
    else:
        model_dir = DEFAULT_MODEL_DIR
    
    # 입력 검증
    # --top_img나 --bottom_img가 지정되면 단일 쌍 처리 모드
    if args.top_img or args.bottom_img:
        # 단일 쌍 처리 모드인데 두 이미지가 모두 지정되지 않음
        if not args.top_img or not args.bottom_img:
            parser.error("--top_img와 --bottom_img를 모두 지정해야 합니다.")
        batch_mode = False
        # 단일 쌍 처리 모드에서는 input_dir 무시 (사용자에게 알림만)
        if args.input_dir and args.input_dir != DEFAULT_INPUT_DIR:
            print(f"  [알림] --top_img/--bottom_img가 지정되어 단일 쌍 처리 모드로 실행됩니다. --input_dir는 무시됩니다.")
    else:
        # 배치 처리 모드 (기본값, 파일명 지정 안 하면 자동으로 배치 처리)
        batch_mode = True
    
    # 분류기 초기화
    classifier = ProcessClassifier(
        model_dir=model_dir,  # config에서 가져온 경로 사용
        device=args.device,
        backbone=args.backbone,
        img_size=args.img_size
    )
    
    # 배치 처리 모드
    if batch_mode:
        print(f"\n배치 처리 모드: {args.input_dir} 폴더에서 이미지 쌍 자동 검색 중...")
        
        # 이미지 쌍 찾기
        pairs = load_image_pairs_from_folder(args.input_dir, max_time_gap_seconds=args.max_time_gap)
        
        if len(pairs) == 0:
            print(f"  경고: {args.input_dir} 폴더에서 이미지 쌍을 찾을 수 없습니다.")
            print(f"  폴더가 존재하는지, 이미지 파일이 있는지 확인해주세요.")
            return None
        
        print(f"  총 {len(pairs)}개 쌍 발견")
        
        # 각 쌍 분류
        save_combined = getattr(args, 'save_combined', True)
        results = []
        process_counts = {'0': 0, '1': 0, 'A': 0}
        
        for idx, (top_path, bottom_path) in enumerate(pairs, 1):
            print(f"\n[{idx}/{len(pairs)}] 분류 중...")
            print(f"  위 이미지: {os.path.basename(top_path)}")
            print(f"  아래 이미지: {os.path.basename(bottom_path)}")
            
            try:
                if not args.no_save:
                    result = classifier.classify_and_save(
                        top_path,
                        bottom_path,
                        args.save_results,
                        save_combined=save_combined
                    )
                else:
                    result = classifier.classify(top_path, bottom_path)
                
                results.append(result)
                process_label = result['process_label']
                process_counts[process_label] += 1
                
                print(f"  공정: {process_label} (신뢰도: {result['confidence']:.4f})")
                
            except Exception as e:
                print(f"  오류: {e}")
                continue
        
        # 전체 결과 요약
        print(f"\n{'='*60}")
        print("배치 처리 완료")
        print(f"{'='*60}")
        print(f"  총 처리: {len(results)}개 쌍")
        print(f"  공정별 분류:")
        for process, count in process_counts.items():
            print(f"    공정 {process}: {count}개")
        
        # JSON 파일로 저장
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_pairs': len(results),
                    'process_counts': process_counts,
                    'results': results
                }, f, indent=2, ensure_ascii=False)
            print(f"\n결과 JSON 저장: {args.output}")
        
        return results
    
    # 단일 쌍 처리 모드
    else:
        print(f"\n이미지 분류 중...")
        print(f"  입력 이미지 1: {args.top_img}")
        print(f"  입력 이미지 2: {args.bottom_img}")
        
        # 결과 저장 (기본적으로 저장, --no_save 옵션이 있으면 저장 안 함)
        if not args.no_save:
            # save_combined 기본값 처리
            save_combined = getattr(args, 'save_combined', True)
            result = classifier.classify_and_save(
                args.top_img, 
                args.bottom_img, 
                args.save_results,
                save_combined=save_combined
            )
            # 순서 보정 여부 확인
            if result.get('order_corrected', False):
                print(f"  [알림] 타임스탬프 기반으로 위/아래 순서 자동 보정됨")
                print(f"    - 위 이미지 (나중): {os.path.basename(result['top_img_path'])}")
                print(f"    - 아래 이미지 (먼저): {os.path.basename(result['bottom_img_path'])}")
            
            print(f"\n분류 결과 저장 완료:")
            print(f"  저장 위치: {args.save_results}/{result['process_label']}/")
            print(f"  쌍 ID: {result['pair_id']}")
            print(f"  저장된 파일:")
            print(f"    - 위 이미지: {os.path.basename(result['saved_files']['top'])}")
            print(f"    - 아래 이미지: {os.path.basename(result['saved_files']['bottom'])}")
            if save_combined and 'combined' in result['saved_files']:
                print(f"    - 결합 이미지: {os.path.basename(result['saved_files']['combined'])}")
            print(f"    - 메타데이터: {os.path.basename(result['saved_files']['metadata'])}")
        else:
            result = classifier.classify(args.top_img, args.bottom_img)
            # 순서 보정 여부 확인
            if result.get('order_corrected', False):
                print(f"  [알림] 타임스탬프 기반으로 위/아래 순서 자동 보정됨")
                print(f"    - 위 이미지 (나중): {os.path.basename(result['top_img_path'])}")
                print(f"    - 아래 이미지 (먼저): {os.path.basename(result['bottom_img_path'])}")
        
        # 결과 출력
        print(f"\n분류 결과:")
        print(f"  공정: {result['process_label']} (번호: {result['process']})")
        print(f"  신뢰도: {result['confidence']:.4f}")
        print(f"  쌍 ID: {result.get('pair_id', 'N/A')}")
        print(f"  확률:")
        for process, prob in result['probabilities'].items():
            print(f"    공정 {process}: {prob:.4f}")
        
        # JSON 파일로 저장
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n결과 JSON 저장: {args.output}")
        
        return result


if __name__ == '__main__':
    main()

