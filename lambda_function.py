import base64
import json
from typing import Any, Dict, List

import cv2
import numpy as np

from infer import DEFAULT_CONFIDENCE_THRESHOLD
from model_loader import MODEL_PATH, get_model

DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
    "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
}


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": DEFAULT_HEADERS,
        "body": json.dumps(payload, default=_to_json_compatible),
    }


def _parse_event_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    body = event.get("body")
    if isinstance(body, dict):
        return body
    if isinstance(body, str) and body.strip():
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}
    return event if isinstance(event, dict) else {}


def _decode_image_from_base64(image_base64: str) -> np.ndarray:
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 image payload") from exc

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Decoded bytes are not a valid image")
    return image


def _serialize_detections(result: Any) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if result.boxes is None:
        return detections

    masks = result.masks
    for i, box in enumerate(result.boxes):
        segment = []
        if masks is not None and i < len(masks.xy):
            segment = masks.xy[i].tolist()

        class_id = int(box.cls[0])
        detections.append(
            {
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": float(box.conf[0]),
                "xyxy": [float(v) for v in box.xyxy[0].tolist()],
                "segment": segment,
            }
        )
    return detections


def _run_inference_from_image(image: np.ndarray, conf: float) -> List[Dict[str, Any]]:
    results = get_model().predict(source=image, device="cpu", conf=conf, verbose=False)
    if not results:
        return []
    return _serialize_detections(results[0])


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        http_method = (event or {}).get("httpMethod", "")
        if http_method == "OPTIONS":
            return _response(200, {"ok": True})

        payload = _parse_event_payload(event or {})
        action = payload.get("action") or (event or {}).get("action") or "predict"

        if action == "health":
            return _response(
                200,
                {
                    "ok": True,
                    "message": "Lambda is healthy",
                    "model_path": MODEL_PATH,
                },
            )

        image_base64 = payload.get("image_base64") or (event or {}).get("image_base64")
        if not image_base64 and (event or {}).get("isBase64Encoded") and isinstance((event or {}).get("body"), str):
            image_base64 = (event or {}).get("body")

        if not image_base64:
            return _response(
                400,
                {
                    "ok": False,
                    "error": "Missing image payload. Provide image_base64 in JSON body or use base64-encoded raw body.",
                },
            )

        conf_raw = payload.get("conf", DEFAULT_CONFIDENCE_THRESHOLD)
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            return _response(400, {"ok": False, "error": "conf must be a number"})

        conf = max(0.0, min(1.0, conf))
        image = _decode_image_from_base64(image_base64)
        detections = _run_inference_from_image(image, conf)

        return _response(
            200,
            {
                "ok": True,
                "count": len(detections),
                "detections": detections,
            },
        )

    except ValueError as exc:
        return _response(400, {"ok": False, "error": str(exc)})
    except Exception as exc:
        return _response(500, {"ok": False, "error": str(exc)})
