from typing import Dict, Any, Optional

import numpy as np
import cv2

def visualize(
    result: Dict[str, Any],         # classify_batch[i]
    save_path: Optional[str] = None,
    alpha: float = 0.5,
    draw_anomaly_map: bool = True,
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
    for region in result["regions"]:
        poly = region["polygon"].astype(np.int32)
        bbox = region["bbox"]
        is_pass = region.get("pass", False)
        color = region.get("color", (0, 0, 0)) if not is_pass else (128, 128, 128)
        a_score = region.get("anomaly_score", 0.0)
        area = region.get("area", 0)

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
        if "class_name" in region and region["class_name"] is not None:
            x1, y1, x2, y2 = bbox
            cls_name = region["class_name"]
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
                2 if not is_pass else 1,
            )

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

    if save_path is not None:
        cv2.imwrite(save_path, vis_img)

    return vis_img
