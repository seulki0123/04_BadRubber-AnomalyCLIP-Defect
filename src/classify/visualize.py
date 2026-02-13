from typing import Dict, Any, Optional

import numpy as np
import cv2

def visualize(
    result: Dict[str, Any],         # classify_batch[i]
    alpha: float = 0.5,
    draw_anomaly_map: bool = True,
    crop_scale: float = 2.0
) -> np.ndarray:
    """
    Visualize anomaly regions + classification results.

    result structure:
    {
        "image": np.ndarray,
        "anomaly_map": np.ndarray,
        "global_anomaly_score": float,
        "regions": [
            {
                "polygon": np.ndarray,
                "bbox": (x1, y1, x2, y2),
                "anomaly_score": float,
                "class_name": str,
                "pass": bool,
                "confidence": float,
            }
        ]
    }
    """
    image = result["image"]
    anomaly_map = result.get("anomaly_map", None)

    vis_img = image.copy()
    crops_img = []

    # ----------------------------
    # anomaly heatmap overlay
    # ----------------------------
    if draw_anomaly_map and anomaly_map is not None:
        amap = np.clip(anomaly_map, 0.0, 1.0)
        heatmap = (amap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        vis_img = cv2.addWeighted(vis_img, 1 - alpha, heatmap, alpha, 0)

    # ----------------------------
    # regions
    # ----------------------------
    for idx, region in enumerate(result["regions"]):
        poly = region["polygon"].astype(np.int32)
        bbox = region["bbox"]
        is_pass = region.get("pass", False)
        color = region.get("color", (0, 0, 0)) if not is_pass else (128, 128, 128)
        a_score = region.get("anomaly_score", 0.0)
        area = region.get("area", 0)

        if is_pass:
            continue

        # polygon
        cv2.polylines(
            vis_img,
            [poly],
            isClosed=True,
            color=color,
            thickness=1,
        )

        # polygon anomaly score
        cx = int(np.clip(poly[:, 0].mean(), 0, vis_img.shape[1] - 1))
        cy = int(np.clip(poly[:, 1].mean(), 0, vis_img.shape[0] - 1))
        cv2.putText(
            vis_img,
            f"A:{a_score:.2f}, {area:.0f}",
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
        )

        # bbox + classification
        x1, y1, x2, y2 = bbox
        cls_name = region.get("class_name", "unknown")
        conf = region.get("confidence", 0.0)

        cv2.rectangle(
            vis_img,
            (x1, y1),
            (x2, y2),
            color,
            1,
        )

        label = f"{cls_name} {conf:.2f}"
        cv2.putText(
            vis_img,
            label,
            (x1, max(y1 - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # ----------------------------
        # crops
        # ----------------------------
        h, w = image.shape[:2]

        x1c = max(0, x1)
        bw = x2 - x1
        bh = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        new_bw = bw * crop_scale
        new_bh = bh * crop_scale

        nx1 = int(cx - new_bw / 2)
        ny1 = int(cy - new_bh / 2)
        nx2 = int(cx + new_bw / 2)
        ny2 = int(cy + new_bh / 2)

        # clamp to image
        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(w, nx2)
        ny2 = min(h, ny2)

        crop = image[ny1:ny2, nx1:nx2]

        crops_img.append({
            "img": crop,
            "cls_name": cls_name,
            "conf": conf,
            "a_score": a_score,
            "area": area,
            "bbox_scaled": (nx1, ny1, nx2, ny2),
        })

    # ----------------------------
    # global anomaly score
    # ----------------------------
    global_score = result.get("global_anomaly_score", None)
    if global_score is not None:
        cv2.putText(
            vis_img,
            f"Global A: {global_score:.3f}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    return vis_img, crops_img