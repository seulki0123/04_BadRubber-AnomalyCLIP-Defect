# -*- coding: utf-8 -*-
"""
baler 분류 서버 설정 (서버 컴퓨터에서 사용)

이 설정 파일은 HTTP 서버가 실행되는 서버 컴퓨터에서만 사용됩니다.
클라이언트 설정은 client_config.py를 참조하세요.

설정 구분:
1. ServerConfig: HTTP 서버가 실행될 설정
   - host: 서버가 바인딩할 IP (0.0.0.0 = 모든 인터페이스에서 접속 가능)
   - port: 서버가 리스닝할 포트

2. ModelPathsConfig: 모델 파일 경로 설정

3. YOLOConfig: YOLO 모델 설정 (불량 검출용)
"""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ServerConfig:
    """서버 설정 (HTTP 서버가 실행될 설정)"""
    host: str = "0.0.0.0"  # 서버가 바인딩할 IP (0.0.0.0 = 모든 인터페이스)
    port: int = 8003
    device: str = "auto"  # cuda, cpu, auto


@dataclass
class ModelPathsConfig:
    """모델 경로 설정"""
    # config.py
    model_path: str = "model/baler/br_b/2550H_paired"
    production: str = "BR-B"
    grade: str = "2550H"


@dataclass
class YOLOConfig:
    """YOLO 모델 설정"""
    enabled: bool = True  # YOLO 불량 검출 사용 여부 (True: 사용, False: 비활성화)
    model_path: str = "model/yolo/ssbr/F3626Y/best_model.pt"  # YOLO 모델 경로 (LG_final 폴더 기준)
    meta_dir: str = "C:\BaleMetadataDB"  # 메타데이터 저장 폴더 (LG_final 폴더 기준)
    faulty_img_dir: str = "faulty_img"  # faulty 이미지 저장 폴더 (LG_final 폴더 기준)
    conf_threshold: float = 0.3  # 신뢰도 임계값
    iou_threshold: float = 0.45  # IoU 임계값
    max_faulty_files: int = 20  # 불량 이미지 최대 보관 개수 (초과 시 오래된 파일 자동 삭제)
    max_meta_files: int = 100000  # 메타데이터 최대 보관 개수 (초과 시 오래된 파일 자동 삭제)
    enable_faulty_cleanup: bool = True  # 불량 이미지 자동 삭제 기능 사용 여부 (True: 사용, False: 비활성화)
    enable_meta_cleanup: bool = True  # 메타데이터 자동 삭제 기능 사용 여부 (True: 사용, False: 비활성화)

@dataclass
class AnomalyInspectorConfig:
    anomalyclip_model_path: str = "model/anomaly_inspector/anomalyclip/9_12_4_mvtec+F1038-F2150-M2520/epoch_15.pth"
    anomalyclip_imgsz: int = 32 * 12

    bgremover_model_path: str = "model/anomaly_inspector/rmbg/SSBR_F2150-M2520-F1038"
    bgremover_imgsz: int = 32 * 5

    classifier_model_path: str = "model/anomaly_inspector/classify/2550H_imgsz32/weights/best.pt"
    classifier_imgsz: int = 32 * 1

    anomaly_threshold: float = 0.25
    anomaly_min_area: int = 112
    classifier_conf_threshold: float = 0.5

@dataclass
class PairMatchingConfig:
    """이미지 쌍 매칭 설정 (700ms 차이 고려)"""
    timeout_seconds: float = 2.0  # part=1과 part=2 매칭 대기 시간 (초) - 여유를 두고 2초로 설정
    cleanup_interval: float = 0.2  # 타임아웃 체크 주기 (초) - 더 자주 체크하여 정확도 향상
    max_pending_pairs: int = 50  # 최대 대기 중인 쌍 개수


@dataclass
class BalerServerConfig:
    """baler 분류 서버 전체 설정"""
    server: ServerConfig = field(default_factory=ServerConfig)  # 서버 실행 설정
    model_paths: ModelPathsConfig = field(default_factory=ModelPathsConfig)  # 모델 경로 설정
    yolo: YOLOConfig = field(default_factory=YOLOConfig)  # YOLO 모델 설정
    pair_matching: PairMatchingConfig = field(default_factory=PairMatchingConfig)  # 이미지 쌍 매칭 설정
    anomaly_inspector: AnomalyInspectorConfig = field(default_factory=AnomalyInspectorConfig)
    
    def get_yolo_model_path(self) -> str:
        """YOLO 모델 절대 경로 반환"""
        base_dir = Path(__file__).parent
        return str(base_dir / self.yolo.model_path)
    
    def get_meta_dir(self, date_str: Optional[str] = None) -> str:
        """메타데이터 저장 폴더 절대 경로 반환
        
        Args:
            date_str: 날짜 문자열 (YYYY-MM-DD 형식, 예: "2025-12-22")
                      None이면 기본 meta 폴더 반환
        """
        base_dir = Path(__file__).parent
        if date_str:
            # 날짜별 폴더 생성: meta/2025-12-22/
            meta_path = base_dir / self.yolo.meta_dir / date_str
        else:
            meta_path = base_dir / self.yolo.meta_dir
        meta_path.mkdir(parents=True, exist_ok=True)
        return str(meta_path)
    
    def get_faulty_img_dir(self, date_str: Optional[str] = None) -> str:
        """faulty 이미지 저장 폴더 절대 경로 반환
        
        Args:
            date_str: 날짜 문자열 (YYYY-MM-DD 형식, 예: "2025-12-22")
                      None이면 기본 faulty_img 폴더 반환
        """
        base_dir = Path(__file__).parent
        if date_str:
            # 날짜별 폴더 생성: faulty_img/2025-12-22/
            faulty_path = base_dir / self.yolo.faulty_img_dir / date_str
        else:
            faulty_path = base_dir / self.yolo.faulty_img_dir
        faulty_path.mkdir(parents=True, exist_ok=True)
        return str(faulty_path)


# 전역 설정 인스턴스
_config: Optional[BalerServerConfig] = None


def get_config() -> BalerServerConfig:
    """전역 설정 인스턴스 반환 (싱글톤 패턴)"""
    global _config
    if _config is None:
        _config = BalerServerConfig()
    return _config


def set_config(config: BalerServerConfig) -> None:
    """전역 설정 인스턴스 설정"""
    global _config
    _config = config

