from ultralytics import YOLO
import base64
import tempfile
from typing import Any, Dict, List

def _run_inference(image_base64: str) -> List[Dict[str, Any]]:
    image_bytes = base64.b64decode(image_base64)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        tmp.write(image_bytes)
        tmp.flush()

        results = _get_model().predict(source=tmp.name, device="cpu", verbose=False)

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            detections.append(
                {
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "xyxy": [float(v) for v in box.xyxy[0].tolist()],
                }
            )
    return detections

def main():

    model = YOLO("yolo26l-seg.pt")

    model.predict(
        source='test_images',
        save=True,
        device=0
    )

if __name__ == "__main__":
    main()