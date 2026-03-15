
from typing import Dict, List, Tuple
import cv2
import numpy as np

from model_loader import get_model


def _segment_to_mask(segment: List[List[float]], image_shape: Tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(segment) < 3:
        return mask.astype(bool)

    pts = np.round(np.array(segment, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
    pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
    pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def run_inference(image_path: str) -> Tuple[cv2.Mat, List[Dict]]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    results = get_model().predict(source=image_path, device="cpu", verbose=False)
    detections = []
    if not results:
        return image, detections
    
    result = results[0]
    if result.boxes is None:
        return image, detections
    masks = result.masks
    for i, box in enumerate(result.boxes):
        segment = []
        pixels = np.zeros(image.shape[:2], dtype=bool)

        if masks is not None and i < len(masks.xy):
            segment = masks.xy[i].tolist()
            pixels = _segment_to_mask(segment, image.shape)
        elif masks is not None and i < len(masks.data):
            # Fallback for malformed polygons: resize mask to original image size.
            raw_mask = masks.data[i].cpu().numpy().astype(np.uint8)
            pixels = cv2.resize(raw_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        detections.append(
            {
                "class_id": int(box.cls[0]),
                "class_name": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "xyxy": [float(v) for v in box.xyxy[0].tolist()],
                "segment": segment,
                "pixels": pixels,
            }
        )
    return image, detections
