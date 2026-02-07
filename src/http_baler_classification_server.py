# -*- coding: utf-8 -*-
"""baler 분류 HTTP 서버 (config에서 모델 경로 설정, YOLO 불량 검출 config로 제어)"""
import argparse
import json
import logging
import os
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from classify_paired_images import ProcessClassifier
from config import get_config
from AnomalyInspector import AnomalyInspector

# YOLO 모델 로드 (ultralytics)
try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None


class F1810BalerClassificationRequestHandler(BaseHTTPRequestHandler):
    """HTTP POST 요청을 통해 이미지 쌍으로 공정 분류를 수행하는 핸들러"""

    classifier: Optional[ProcessClassifier] = None  # 분류기 (싱글톤)
    classifier_lock: threading.Lock = threading.Lock()  # 분류기 로드 락
    processing_lock: threading.Lock = threading.Lock()
    yolo_model: Optional[Any] = None  # YOLO 모델 (싱글톤)
    yolo_model_lock: threading.Lock = threading.Lock()  # YOLO 모델 로드 락
    config: Optional[Any] = None  # 설정
    current_baler_value: Optional[str] = None  # 현재 baler 값 (단일 변수)
    baler_lock: threading.Lock = threading.Lock()  # baler 값 접근 락

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        """JSON 응답 전송"""
        client_ip = self.client_address[0] if self.client_address else "unknown"
        
        # 에러 응답인 경우 로깅
        if status_code >= 400:
            error_msg = payload.get("message", "알 수 없는 오류")
            logging.warning(f"[{client_ip}] {status_code} 에러 응답: {error_msg}")
        
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError) as e:
            # 클라이언트가 연결을 끊은 경우 (타임아웃 등)
            logging.warning(f"클라이언트 연결이 끊어졌습니다: {e}")
            return

    def _read_json_body(self) -> Optional[Dict[str, Any]]:
        """JSON 요청 본문 읽기"""
        client_ip = self.client_address[0] if self.client_address else "unknown"
        
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            logging.warning(f"[{client_ip}] Content-Length가 0이거나 없음")
            return None
        
        raw_body = self.rfile.read(content_length)
        if not raw_body:
            logging.warning(f"[{client_ip}] 요청 본문이 비어있음")
            return None
        
        decoded_body = None
        try:
            decoded_body = raw_body.decode("utf-8")
            return json.loads(decoded_body)
        except UnicodeDecodeError as e:
            logging.warning(f"[{client_ip}] 요청 본문 디코딩 실패 (UTF-8): {e}, 본문 일부: {raw_body[:100]}")
            return None
        except json.JSONDecodeError as e:
            body_preview = decoded_body[:200] if decoded_body else raw_body[:200]
            logging.warning(f"[{client_ip}] JSON 파싱 실패: {e}, 받은 본문: {body_preview}")
            return None

    def _get_classifier(self, device: str = "auto") -> Optional[ProcessClassifier]:
        """분류기 가져오기 (싱글톤 패턴)"""
        # 이미 로드된 분류기가 있으면 반환
        if self.classifier is not None:
            return self.classifier
        
        # 락을 사용하여 동시 로드 방지
        with self.classifier_lock:
            # 다시 확인 (다른 스레드가 이미 로드했을 수 있음)
            if self.classifier is not None:
                return self.classifier
            
            # config에서 모델 경로 가져오기
            if self.config is None:
                logging.error("설정이 로드되지 않았습니다.")
                return None
            
            model_dir = self.config.model_paths.model_path
            if not os.path.exists(model_dir):
                logging.error(f"모델 디렉토리가 없습니다: {model_dir}")
                return None
            
            try:
                # num_classes=None으로 설정하여 체크포인트에서 자동 감지
                classifier = ProcessClassifier(
                    model_dir=model_dir,
                    device=device,
                    backbone="resnet18",
                    img_size=224,
                    num_classes=None  # 자동 감지
                )
                self.classifier = classifier
                # 자동 감지된 클래스 수와 레이블 로깅
                detected_classes = classifier.num_classes if hasattr(classifier, 'num_classes') else 'unknown'
                detected_labels = classifier.process_labels if hasattr(classifier, 'process_labels') else 'unknown'
                production = self.config.model_paths.production
                grade = self.config.model_paths.grade
                logging.info(f"{production} {grade} 분류기 로드 완료: {model_dir} (자동 감지된 클래스 수: {detected_classes}, 레이블: {detected_labels})")
                return classifier
            except Exception as e:
                logging.error(f"분류기 로드 실패: {e}")
                return None
    
    def _convert_to_json_serializable(self, obj):
        """numpy 배열과 기타 JSON 직렬화 불가능한 객체를 변환 (최적화)"""
        import numpy as np
        
        # 빠른 경로: 이미 JSON 직렬화 가능한 타입
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        
        # numpy 타입 처리
        if isinstance(obj, np.ndarray):
            # 작은 배열만 리스트로 변환, 큰 배열은 스킵하거나 요약
            if obj.size > 10000:  # 큰 배열은 스킵
                return f"<array shape={obj.shape} dtype={obj.dtype}>"
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        
        # 컬렉션 타입 처리
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._convert_to_json_serializable(item) for item in obj)
        
        return obj
    
    def _get_yolo_model(self) -> Optional[Any]:
        """YOLO 모델 가져오기 (싱글톤 패턴)"""
        if not YOLO_AVAILABLE:
            logging.warning("YOLO가 설치되지 않았습니다. 불량 검출 기능을 사용할 수 없습니다.")
            return None
        
        if self.config is None:
            logging.error("설정이 로드되지 않았습니다.")
            return None
        
        # config에서 YOLO 활성화 여부 확인
        if not self.config.yolo.enabled:
            return None
        
        # 이미 로드된 모델이 있으면 반환
        if self.yolo_model is not None:
            return self.yolo_model
        
        # 락을 사용하여 동시 로드 방지
        with self.yolo_model_lock:
            # 다시 확인 (다른 스레드가 이미 로드했을 수 있음)
            if self.yolo_model is not None:
                return self.yolo_model
            
            try:
                yolo_model_path = self.config.get_yolo_model_path()
                if not os.path.exists(yolo_model_path):
                    logging.error(f"YOLO 모델 파일을 찾을 수 없습니다: {yolo_model_path}")
                    return None
                
                self.yolo_model = YOLO(yolo_model_path)
                logging.info(f"YOLO 모델 로드 완료: {yolo_model_path}")
                return self.yolo_model
            except Exception as e:
                logging.error(f"YOLO 모델 로드 실패: {e}")
                return None
    
    def _detect_faulty_spots(self, image_path: str) -> Tuple[Optional[Any], List[Dict[str, Any]], bool]:
        """
        이미지에서 불량(wet spot) 검출
        
        Args:
            image_path: 이미지 경로
            
        Returns:
            (annotated_image, detections, has_faulty): 검출된 이미지, 검출 정보 리스트, 불량 여부
        """
        if not YOLO_AVAILABLE:
            logging.warning("YOLO가 사용 불가능합니다.")
            return None, [], False
        
        yolo_model = self._get_yolo_model()
        if yolo_model is None:
            logging.warning("YOLO 모델을 로드할 수 없습니다.")
            return None, [], False
        
        try:
            if self.config is None:
                logging.warning("config가 None입니다.")
                return None, [], False
            
            logging.info(f"YOLO 검출 시작: {image_path}, conf_threshold={self.config.yolo.conf_threshold}, iou_threshold={self.config.yolo.iou_threshold}")
            
            # YOLOv8-seg 추론 수행
            results = yolo_model.predict(
                source=image_path,
                conf=self.config.yolo.conf_threshold,
                iou=self.config.yolo.iou_threshold,
                save=False,
                verbose=False,
                task='segment'  # segmentation 모드
            )
            
            result = results[0]
            detections = []
            has_faulty = False
            
            # 검출 결과 확인
            boxes_count = len(result.boxes) if result.boxes is not None else 0
            masks_count = len(result.masks) if result.masks is not None else 0
            
            logging.info(f"YOLO 검출 결과: boxes={boxes_count}, masks={masks_count}")
            
            # 검출 결과 확인 (빠른 경로: 검출이 없으면 즉시 반환)
            if (result.boxes is None or len(result.boxes) == 0) and \
               (result.masks is None or len(result.masks) == 0):
                logging.info(f"검출 없음: {image_path}")
                return None, [], False
            
            has_faulty = True
            logging.info(f"불량 검출됨: boxes={boxes_count}, masks={masks_count}, image={image_path}")
            
            # 원본 이미지 로드 (BGR 형식)
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"이미지를 로드할 수 없습니다: {image_path}")
                return None, [], False
            
            # 원본 이미지 복사 (주석 그리기용)
            annotated_image = image.copy()
            
            # 클래스 이름 가져오기
            class_names = yolo_model.names if hasattr(yolo_model, 'names') else {}
            
            # 박스 그리기
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 클래스 이름 가져오기
                    class_name = class_names.get(cls_id, f"class_{cls_id}")
                    
                    # 박스 그리기
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    
                    # 텍스트 크기 계산 (이미지 크기에 비례)
                    img_h, img_w = annotated_image.shape[:2]
                    font_scale = max(0.5, min(img_w, img_h) / 1000.0)
                    thickness = max(1, int(font_scale * 2))
                    
                    # 텍스트 배경 그리기
                    label = f"{class_name} {conf:.2f}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    cv2.rectangle(annotated_image, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 255), -1)
                    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
                    
                    # 검출 정보 저장
                    detections.append({
                        "class": int(cls_id),
                        "class_name": class_name,
                        "confidence": float(conf),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })
            
            # 마스크 그리기 (최적화: 모든 마스크를 한 번에 처리)
            if result.masks is not None and len(result.masks) > 0:
                masks = result.masks.data.cpu().numpy()
                
                # 모든 마스크를 한 번에 처리 (이미지 복사 최소화)
                combined_mask = np.zeros((annotated_image.shape[0], annotated_image.shape[1]), dtype=np.uint8)
                
                for mask in masks:
                    # 마스크를 원본 이미지 크기로 리사이즈
                    mask_resized = cv2.resize(mask, (annotated_image.shape[1], annotated_image.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    # 마스크 합치기
                    combined_mask = np.maximum(combined_mask, mask_binary)
                
                # 한 번만 오버레이 처리
                if np.any(combined_mask > 0):
                    overlay = annotated_image.copy()
                    overlay[combined_mask > 0] = [0, 255, 255]  # 노란색 (BGR)
                    cv2.addWeighted(overlay, 0.5, annotated_image, 0.5, 0, annotated_image)
            
            logging.info(f"불량 검출 완료: {len(detections)}개 검출, image={image_path}")
            return annotated_image, detections, has_faulty
            
        except Exception as e:
            logging.error(f"불량 검출 중 오류 발생: {e}", exc_info=True)
            return None, [], False
    
    def _detect_faulty_spots_batch(self, image_paths: List[str]) -> List[Tuple[Optional[Any], List[Dict[str, Any]], bool]]:
        """
        여러 이미지를 배치로 YOLO 검출
        
        Args:
            image_paths: 이미지 경로 리스트
        
        Returns:
            [(annotated_image, detections, has_faulty), ...] 리스트
        """
        if not YOLO_AVAILABLE:
            logging.warning("YOLO가 사용 불가능합니다.")
            return [(None, [], False)] * len(image_paths)
        
        yolo_model = self._get_yolo_model()
        if yolo_model is None:
            logging.warning("YOLO 모델을 로드할 수 없습니다.")
            return [(None, [], False)] * len(image_paths)
        
        try:
            if self.config is None:
                logging.warning("config가 None입니다.")
                return [(None, [], False)] * len(image_paths)
            
            logging.info(f"YOLO 배치 검출 시작: {len(image_paths)}장, conf_threshold={self.config.yolo.conf_threshold}, iou_threshold={self.config.yolo.iou_threshold}")
            
            # YOLOv8-seg 배치 추론 수행
            results = yolo_model.predict(
                source=image_paths,  # 리스트로 전달하면 배치 처리
                conf=self.config.yolo.conf_threshold,
                iou=self.config.yolo.iou_threshold,
                save=False,
                verbose=False,
                task='segment'
            )
            
            batch_results = []
            class_names = yolo_model.names if hasattr(yolo_model, 'names') else {}
            
            for idx, (result, image_path) in enumerate(zip(results, image_paths)):
                detections = []
                has_faulty = False
                
                # 검출 결과 확인
                boxes_count = len(result.boxes) if result.boxes is not None else 0
                masks_count = len(result.masks) if result.masks is not None else 0
                
                if (result.boxes is None or len(result.boxes) == 0) and \
                   (result.masks is None or len(result.masks) == 0):
                    batch_results.append((None, [], False))
                    continue
                
                has_faulty = True
                logging.info(f"불량 검출됨 [{idx+1}/{len(image_paths)}]: boxes={boxes_count}, masks={masks_count}, image={os.path.basename(image_path)}")
                
                # 원본 이미지 로드
                image = cv2.imread(image_path)
                if image is None:
                    logging.error(f"이미지를 로드할 수 없습니다: {image_path}")
                    batch_results.append((None, [], False))
                    continue
                
                annotated_image = image.copy()
                
                # 박스 그리기
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = map(int, box)
                        class_name = class_names.get(cls_id, f"class_{cls_id}")
                        
                        # 박스 그리기
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        
                        # 텍스트 크기 계산
                        img_h, img_w = annotated_image.shape[:2]
                        font_scale = max(0.5, min(img_w, img_h) / 1000.0)
                        thickness = max(1, int(font_scale * 2))
                        
                        # 텍스트 배경 그리기
                        label = f"{class_name} {conf:.2f}"
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        cv2.rectangle(annotated_image, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 255), -1)
                        cv2.putText(annotated_image, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
                        
                        # 검출 정보 저장
                        detections.append({
                            "class": int(cls_id),
                            "class_name": class_name,
                            "confidence": float(conf),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })
                
                # 마스크 그리기
                if result.masks is not None and len(result.masks) > 0:
                    masks = result.masks.data.cpu().numpy()
                    combined_mask = np.zeros((annotated_image.shape[0], annotated_image.shape[1]), dtype=np.uint8)
                    
                    for mask in masks:
                        mask_resized = cv2.resize(mask, (annotated_image.shape[1], annotated_image.shape[0]))
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                        combined_mask = np.maximum(combined_mask, mask_binary)
                    
                    if np.any(combined_mask > 0):
                        overlay = annotated_image.copy()
                        overlay[combined_mask > 0] = [0, 255, 255]
                        cv2.addWeighted(overlay, 0.5, annotated_image, 0.5, 0, annotated_image)
                
                batch_results.append((annotated_image, detections, has_faulty))
            
            logging.info(f"YOLO 배치 검출 완료: {len([r for r in batch_results if r[2]])}장에서 불량 검출")
            return batch_results
            
        except Exception as e:
            logging.error(f"YOLO 배치 검출 실패: {e}", exc_info=True)
            return [(None, [], False)] * len(image_paths)
    
    def _extract_date_from_id(self, id_str: str) -> Optional[str]:
        """ID 문자열에서 날짜 추출하여 YYYY-MM-DD 형식으로 반환
        
        Args:
            id_str: ID 문자열 (예: "251222_002800")
        
        Returns:
            날짜 문자열 (예: "2025-12-22") 또는 None
        """
        import re
        # YYMMDD 형식 추출 (예: 251222)
        match = re.match(r'(\d{2})(\d{2})(\d{2})_', id_str)
        if match:
            yy, mm, dd = match.groups()
            year = 2000 + int(yy)  # 25 -> 2025
            return f"{year}-{mm}-{dd}"
        return None
    
    def _save_faulty_image(self, annotated_image: Any, original_image_path: str, id_str: str) -> Optional[str]:
        """
        불량 검출된 이미지 저장
        
        Args:
            annotated_image: 검출 결과가 그려진 이미지
            original_image_path: 원본 이미지 경로
            id_str: ID 문자열
            
        Returns:
            저장된 이미지 절대 경로 (None이면 실패)
        """
        if self.config is None:
            logging.warning("config가 None입니다. 불량 이미지를 저장할 수 없습니다.")
            return None
        
        try:
            # 이미지 데이터 확인
            if annotated_image is None:
                logging.error("annotated_image가 None입니다. 불량 이미지를 저장할 수 없습니다.")
                return None
            
            # ID에서 날짜 추출
            date_str = self._extract_date_from_id(id_str)
            
            faulty_img_dir = self.config.get_faulty_img_dir(date_str=date_str)
            logging.info(f"faulty_img_dir: {faulty_img_dir} (날짜: {date_str})")
            
            # 디렉토리 생성
            os.makedirs(faulty_img_dir, exist_ok=True)
            
            # 디렉토리 존재 확인
            if not os.path.exists(faulty_img_dir):
                logging.error(f"faulty_img_dir 생성 실패: {faulty_img_dir}")
                return None
            
            # 파일명 생성: 원본파일명_타임스탬프.jpg
            original_filename = os.path.basename(original_image_path)
            name_without_ext = os.path.splitext(original_filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            faulty_filename = f"{name_without_ext}_{timestamp}.jpg"
            faulty_image_path = os.path.join(faulty_img_dir, faulty_filename)
            
            logging.info(f"불량 이미지 저장 시도: {faulty_image_path}")
            
            # 이미지 저장 (이미 BGR 형식이므로 그대로 저장)
            success = cv2.imwrite(faulty_image_path, annotated_image)
            
            if not success:
                logging.error(f"cv2.imwrite 실패: {faulty_image_path}")
                logging.error(f"  이미지 shape: {annotated_image.shape if hasattr(annotated_image, 'shape') else 'unknown'}")
                logging.error(f"  이미지 dtype: {annotated_image.dtype if hasattr(annotated_image, 'dtype') else 'unknown'}")
                return None
            
            # 파일 존재 확인
            if not os.path.exists(faulty_image_path):
                logging.error(f"이미지 파일이 생성되지 않았습니다: {faulty_image_path}")
                return None
            
            logging.info(f"불량 이미지 저장 성공: {faulty_image_path}")
            
            # 파일 개수 확인 및 자동 삭제 (config에서 설정한 개수 초과 시)
            if self.config and self.config.yolo.enable_faulty_cleanup:
                max_files = self.config.yolo.max_faulty_files if self.config else 20
                self._cleanup_old_files(faulty_img_dir, max_files=max_files)
            
            return os.path.abspath(faulty_image_path)
        except Exception as e:
            logging.error(f"불량 이미지 저장 실패: {e}", exc_info=True)
            return None
    
    def _save_metadata(self, id_str: str, side: int, baler_value: Optional[str], 
                       detections: List[Dict[str, Any]], up_image_path: Optional[str], 
                       down_image_path: str, faulty_img_path: Optional[str],
                       up_faulty_img_path: Optional[str] = None,
                       up_detections: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """
        메타데이터 JSON 파일 저장
        
        Args:
            id_str: ID 문자열
            side: side 값
            baler_value: baler 값 (None이면 분류하지 않음)
            detections: 아래 이미지 불량 검출 정보 리스트
            up_image_path: 위 이미지 경로 (None 가능)
            down_image_path: 아래 이미지 경로
            faulty_img_path: 아래 이미지 불량 이미지 경로 (None 가능)
            up_faulty_img_path: 위 이미지 불량 이미지 경로 (None 가능)
            up_detections: 위 이미지 불량 검출 정보 리스트 (None 가능)
            
        Returns:
            저장된 메타데이터 파일 절대 경로 (None이면 실패)
        """
        if self.config is None:
            return None
        
        try:
            # ID에서 날짜 추출
            date_str = self._extract_date_from_id(id_str)
            
            meta_dir = self.config.get_meta_dir(date_str=date_str)
            os.makedirs(meta_dir, exist_ok=True)
            
            # config에서 production과 grade 가져오기
            production = self.config.model_paths.production if self.config else "UNKNOWN"
            grade = self.config.model_paths.grade if self.config else "UNKNOWN"
            
            # 파일명 생성: {production}_{grade}_{id}_{side}_{타임스탬프}.json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            meta_filename = f"{production}_{grade}_{id_str}_{side}_{timestamp}.json".replace("-", "_")
            meta_path = os.path.join(meta_dir, meta_filename)
            
            # 위/아래 검출 결과 합치기
            all_detections = detections.copy()
            if up_detections:
                all_detections.extend(up_detections)
            
            # 메타데이터 구성 (numpy 배열을 JSON 직렬화 가능한 형태로 변환)
            metadata = {
                "id": id_str,
                "production": production,
                "grade": grade,
                "side": side,
                "baler": baler_value,
                "down_image_path": down_image_path,
                "faulty_img_path": faulty_img_path,
                "detections": self._convert_to_json_serializable(all_detections),
                "detection_count": len(all_detections),
                "timestamp": timestamp
            }
            
            if up_image_path:
                metadata["up_image_path"] = up_image_path
            
            # 위 이미지 검출 결과 추가
            if up_faulty_img_path:
                metadata["up_faulty_img_path"] = up_faulty_img_path
            if up_detections:
                metadata["up_detections"] = self._convert_to_json_serializable(up_detections)
                metadata["up_detection_count"] = len(up_detections)
            
            # 아래 이미지 검출 결과도 별도로 저장
            if detections:
                metadata["down_detections"] = self._convert_to_json_serializable(detections)
                metadata["down_detection_count"] = len(detections)
            
            # JSON 파일 저장
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # 파일 개수 확인 및 자동 삭제 (config에서 설정한 개수 초과 시)
            if self.config and self.config.yolo.enable_meta_cleanup:
                max_files = self.config.yolo.max_meta_files if self.config else 20
                self._cleanup_old_files(meta_dir, max_files=max_files)
            
            return os.path.abspath(meta_path)
        except Exception as e:
            logging.error(f"메타데이터 저장 실패: {e}")
            return None
    
    def _save_multi_side_metadata(self, id_str: str, baler_value: Optional[str],
                                  images_data: List[Dict[str, Any]]) -> Optional[str]:
        """
        다중 side 이미지 메타데이터 JSON 파일 저장 (6장 정보)
        
        Args:
            id_str: ID 문자열
            baler_value: baler 값 (side=1의 part=1,2로 분류한 결과)
            images_data: 이미지 정보 리스트 [{"side": 1, "part": 1, "image_path": "...", "faulty_img_path": "...", "detections": [...]}, ...]
            
        Returns:
            저장된 메타데이터 파일 절대 경로 (None이면 실패)
        """
        if self.config is None:
            return None
        
        try:
            # ID에서 날짜 추출
            date_str = self._extract_date_from_id(id_str)
            
            meta_dir = self.config.get_meta_dir(date_str=date_str)
            os.makedirs(meta_dir, exist_ok=True)
            
            # config에서 production과 grade 가져오기
            production = self.config.model_paths.production if self.config else "UNKNOWN"
            grade = self.config.model_paths.grade if self.config else "UNKNOWN"
            
            # 파일명 생성: {production}_{grade}_{id}_{타임스탬프}.json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            meta_filename = f"{production}_{grade}_{id_str}_{timestamp}.json".replace("-", "_")
            meta_path = os.path.join(meta_dir, meta_filename)
            
            # 메타데이터 구성
            metadata = {
                "id": id_str,
                "production": production,
                "grade": grade,
                "baler": baler_value,
                "timestamp": timestamp,
                "images": []
            }
            
            total_detection_count = 0
            for img_data in images_data:
                side = img_data.get("side")
                part = img_data.get("part")
                image_path = img_data.get("image_path")
                faulty_img_path = img_data.get("faulty_img_path")
                detections = img_data.get("detections", [])
                
                image_meta = {
                    "side": side,
                    "part": part,
                    "image_path": image_path,
                    "faulty_img_path": faulty_img_path,
                    "detections": self._convert_to_json_serializable(detections),
                    "detection_count": len(detections)
                }
                
                metadata["images"].append(image_meta)
                total_detection_count += len(detections)
            
            metadata["total_detection_count"] = total_detection_count
            
            # JSON 파일 저장
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # 파일 개수 확인 및 자동 삭제
            if self.config and self.config.yolo.enable_meta_cleanup:
                max_files = self.config.yolo.max_meta_files if self.config else 20
                self._cleanup_old_files(meta_dir, max_files=max_files)
            
            return os.path.abspath(meta_path)
        except Exception as e:
            logging.error(f"다중 side 메타데이터 저장 실패: {e}", exc_info=True)
            return None
    
    def _cleanup_old_files(self, directory: str, max_files: int = 20):
        """
        디렉토리 내 파일 개수가 max_files를 초과하면 오래된 파일 삭제
        
        Args:
            directory: 디렉토리 경로
            max_files: 최대 파일 개수
        """
        try:
            if not os.path.exists(directory):
                return
            
            files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            if len(files) <= max_files:
                return
            
            # 파일을 수정 시간으로 정렬 (오래된 것부터)
            file_paths = [os.path.join(directory, f) for f in files]
            file_paths.sort(key=lambda x: os.path.getmtime(x))
            
            # 오래된 파일 삭제
            files_to_delete = file_paths[:len(files) - max_files]
            for filepath in files_to_delete:
                try:
                    os.remove(filepath)
                    logging.info(f"오래된 파일 삭제: {filepath}")
                except Exception as e:
                    logging.warning(f"파일 삭제 실패: {filepath}, 오류: {e}")
        except Exception as e:
            logging.error(f"파일 정리 중 오류 발생: {e}")

    def _handle_multi_side_classify(self) -> None:
        """side 1,2,3의 part 1,2 이미지 6장 또는 추가 3장 처리"""
        client_ip = self.client_address[0] if self.client_address else "unknown"
        
        payload = self._read_json_body()
        if payload is None:
            self._send_json(400, {"status": "error", "message": "Invalid JSON"})
            return
        
        logging.info(f"[{client_ip}] 받은 요청 데이터: {json.dumps(payload, ensure_ascii=False)}")
        
        # 필수 필드 확인
        id_value = payload.get("id")
        if not id_value:
            self._send_json(400, {"status": "error", "message": "'id' 필드가 필요합니다."})
            return
        
        images = payload.get("images", {})
        if not isinstance(images, dict):
            self._send_json(400, {"status": "error", "message": "'images' 객체가 필요합니다."})
            return
        
        id_str = str(id_value)
        production = self.config.model_paths.production if self.config else "UNKNOWN"
        
        # side 1,2,3 또는 side 4,5,6 이미지 경로 수집
        image_paths = []
        image_info = []  # (side, part, image_path) 튜플 리스트
        
        # 요청에 포함된 모든 side를 동적으로 찾기
        side_numbers = []
        for key in images.keys():
            if key.startswith("side") and key[4:].isdigit():
                side_num = int(key[4:])
                if 1 <= side_num <= 6:
                    side_numbers.append(side_num)
        
        if not side_numbers:
            self._send_json(400, {"status": "error", "message": "최소 1개 이상의 side가 필요합니다."})
            return
        
        side_numbers.sort()  # 정렬
        
        # side 1-3인지 4-6인지 확인
        # side 1-3인 경우: 모든 side가 1,2,3 중 하나이고, 1,2,3이 모두 포함되어야 함
        # side 4-6인 경우: 모든 side가 4,5,6 중 하나이고, 4,5,6이 모두 포함되어야 함
        is_side_1_3 = all(side in [1, 2, 3] for side in side_numbers) and set(side_numbers) == {1, 2, 3}
        is_side_4_6 = all(side in [4, 5, 6] for side in side_numbers) and set(side_numbers) == {4, 5, 6}
        
        if not (is_side_1_3 or is_side_4_6):
            if all(side in [1, 2, 3] for side in side_numbers):
                self._send_json(400, {"status": "error", "message": "side 1,2,3이 모두 필요합니다."})
            elif all(side in [4, 5, 6] for side in side_numbers):
                self._send_json(400, {"status": "error", "message": "side 4,5,6이 모두 필요합니다."})
            else:
                self._send_json(400, {"status": "error", "message": "side는 1-3 또는 4-6만 허용됩니다."})
            return
        
        # 각 side별로 이미지 경로 수집
        for side_num in side_numbers:
            side_key = f"side{side_num}"
            side_data = images.get(side_key, {})
            
            if not isinstance(side_data, dict):
                self._send_json(400, {"status": "error", "message": f"'{side_key}' 객체가 필요합니다."})
                return
            
            # side 1-3인 경우: part1, part2 모두 필요
            # side 4-6인 경우: part1만 필요
            if is_side_1_3:
                part_numbers = [1, 2]
            else:  # side 4-6
                part_numbers = [1]
            
            for part_num in part_numbers:
                part_key = f"part{part_num}"
                image_path = side_data.get(part_key)
                
                if not image_path or not isinstance(image_path, str):
                    self._send_json(400, {"status": "error", "message": f"'{side_key}.{part_key}' 이미지 경로가 필요합니다."})
                    return
                
                if not os.path.exists(image_path):
                    self._send_json(404, {"status": "error", "message": f"이미지 파일을 찾을 수 없습니다: {image_path}"})
                    return
                
                image_paths.append(image_path)
                image_info.append((side_num, part_num, image_path))
        
        # 이미지 개수 확인
        image_count = len(image_paths)
        
        if image_count == 0:
            self._send_json(400, {"status": "error", "message": "이미지가 없습니다."})
            return
        
        # side 1-3인 경우: 6장 (3 side × 2 part)
        # side 4-6인 경우: 3장 (3 side × 1 part)
        expected_count = 6 if is_side_1_3 else 3
        if image_count != expected_count:
            self._send_json(400, {"status": "error", "message": f"이미지는 {expected_count}장이어야 합니다. (현재: {image_count}장)"})
            return
        
        # 1단계: baler 분류 (side 1-3인 경우만 수행)
        baler_value = None
        if is_side_1_3:
            # side1의 part1, part2로 baler 분류
            side1_part1_path = None
            side1_part2_path = None
            for side, part, path in image_info:
                if side == 1:
                    if part == 1:
                        side1_part1_path = path
                    elif part == 2:
                        side1_part2_path = path
            
            if side1_part1_path and side1_part2_path:
                try:
                    device = self.server.device if hasattr(self.server, 'device') else "auto"
                    if device == "auto":
                        try:
                            import torch
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                        except ImportError:
                            device = "cpu"
                    
                    classifier = self._get_classifier(device=device)
                    if classifier:
                        result = classifier.classify(side1_part1_path, side1_part2_path)
                        baler_value = result['process_label']
                        logging.info(f"[{client_ip}] baler 분류 결과: {baler_value}")
                except Exception as e:
                    logging.error(f"[{client_ip}] baler 분류 실패: {e}")
        else:
            # side 4-6인 경우: baler 분류 안 함
            baler_value = None
            logging.info(f"[{client_ip}] side 4-6 요청: baler 분류 생략")
        
        # 2단계: 모든 이미지를 YOLO 배치 처리
        yolo_results = []
        if self.config and self.config.yolo.enabled:
            yolo_results = self.anomaly_inspector.detect_faulty_spots_batch(image_paths)
        else:
            yolo_results = [(None, [], False)] * image_count
        
        # 3단계: 결과 통합 및 저장
        images_data = []
        response_images = []
        
        for idx, ((side, part, image_path), (annotated_img, detections, has_faulty)) in enumerate(zip(image_info, yolo_results)):
            faulty_img_path = None
            if has_faulty and annotated_img is not None:
                faulty_img_path = self._save_faulty_image(annotated_img, image_path, f"{id_str}_s{side}_p{part}")
            
            # 응답용 이미지 정보
            response_images.append({
                "side": side,
                "part": part,
                "image_path": image_path,
                "faulty_img_path": faulty_img_path
            })
            
            # 메타데이터용 이미지 정보
            images_data.append({
                "side": side,
                "part": part,
                "image_path": image_path,
                "faulty_img_path": faulty_img_path,
                "detections": detections
            })
        
        # 메타데이터 저장
        meta_path = self._save_multi_side_metadata(id_str, baler_value, images_data)
        
        # 응답 반환
        response = {
            "status": "success",
            "data": {
                "id": id_str,
                "production": production,
                "baler": baler_value,
                "images": response_images,
                "meta_path": meta_path
            }
        }
        
        response_serializable = self._convert_to_json_serializable(response)
        logging.info(f"[{client_ip}] 응답 데이터: {json.dumps(response_serializable, ensure_ascii=False)}")
        
        self._send_json(200, response_serializable)

    def do_POST(self) -> None:  # noqa: N802
        """POST 요청 처리"""
        parsed = urlparse(self.path)
        if parsed.path == "/classify_multi_side":
            self._handle_multi_side_classify()
        else:
            self._send_json(404, {"status": "error", "message": "Not found"})

    def do_GET(self) -> None:  # noqa: N802
        """GET 요청 처리"""
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/health"}:
            classifier_status = "loaded" if self.classifier is not None else "not_loaded"
            yolo_status = "loaded" if self.yolo_model is not None else "not_loaded"
            self._send_json(200, {
                "status": "ok",
                "classifier": classifier_status,
                "yolo_model": yolo_status
            })
        else:
            self._send_json(404, {"status": "error"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        """로그 메시지 출력"""
        logging.info("%s - %s", self.address_string(), format % args)



def parse_args() -> argparse.Namespace:
    """명령줄 인수 파싱"""
    # config에서 기본값 가져오기
    config = get_config()
    
    parser = argparse.ArgumentParser(
        description="HTTP baler 분류 서버 (config에서 모델 경로 설정, YOLO 불량 검출 config로 제어)"
    )
    parser.add_argument("--host", default=config.server.host, help=f"서버 바인딩 호스트 (기본: {config.server.host})")
    parser.add_argument("--port", type=int, default=config.server.port, help=f"서버 포트 (기본: {config.server.port})")
    parser.add_argument(
        "--device",
        default=config.server.device,
        help=f"추론 디바이스 (cuda/cpu/auto, 기본: {config.server.device})"
    )
    return parser.parse_args()


def main() -> None:
    """메인 함수"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s"
    )
    args = parse_args()

    # 설정 로드
    config = get_config()
    logging.info("설정 로드 완료")

    # 디바이스 설정
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # 핸들러에 config 설정
    F1810BalerClassificationRequestHandler.config = config
    F1810BalerClassificationRequestHandler.anomaly_inspector = AnomalyInspector(
        anomalyclip_checkpoint_path=config.anomaly_inspector.anomalyclip_model_path,
        bgremover_checkpoint_path=config.anomaly_inspector.bgremover_model_path,
        classifier_checkpoint_path=config.anomaly_inspector.classifier_model_path,
        anomalyclip_imgsz=config.anomaly_inspector.anomalyclip_imgsz,
        bgremover_imgsz=config.anomaly_inspector.bgremover_imgsz,
        classifier_imgsz=config.anomaly_inspector.classifier_imgsz,
        anomaly_threshold=config.anomaly_inspector.anomaly_threshold,
        anomaly_min_area=config.anomaly_inspector.anomaly_min_area,
        classifier_conf_threshold=config.anomaly_inspector.classifier_conf_threshold,
    )

    # 서버에 device 정보 저장
    class CustomHTTPServer(ThreadingHTTPServer):
        def __init__(self, *args, **kwargs):
            self.device = device
            super().__init__(*args, **kwargs)

    server_address = (args.host, args.port)
    httpd = CustomHTTPServer(server_address, F1810BalerClassificationRequestHandler)

    production = config.model_paths.production
    grade = config.model_paths.grade
    model_path = config.model_paths.model_path
    
    logging.info("HTTP baler 분류 서버 시작: http://%s:%s", args.host, args.port)
    logging.info("  - POST /classify_multi_side: 다중 side 이미지 분류 (side 1,2,3의 part 1,2 이미지 6장)")
    logging.info("  - GET /health: 서버 상태 확인")
    logging.info("  - Production: %s", production)
    logging.info("  - Grade: %s", grade)
    logging.info("  - 모델 경로: %s", model_path)
    yolo_status = "활성화" if config.yolo.enabled else "비활성화"
    yolo_model_path = config.yolo.model_path if config.yolo.enabled else "N/A"
    logging.info("  - YOLO 불량 검출: %s (모델: %s)", yolo_status, yolo_model_path)
    logging.info("  - 디바이스: %s", device)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logging.info("서버 종료 중...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

