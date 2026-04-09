import base64
import json
import math
from typing import Any, Dict, List

import cv2
import numpy as np

from color_logic import DEFAULT_LAB_CONFIG_PATH, HoldColor, LabColorClassifier
from infer import DEFAULT_CONFIDENCE_THRESHOLD
from model_loader import MODEL_PATH, get_model

DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
    "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
}

_color_classifier: LabColorClassifier | None = None


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


def _get_color_classifier() -> LabColorClassifier:
    global _color_classifier
    if _color_classifier is None:
        _color_classifier = LabColorClassifier.from_config(DEFAULT_LAB_CONFIG_PATH)
    return _color_classifier


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
                "id": i,
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


def _annotate_detection_colors(image: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    classifier = _get_color_classifier()
    for det in detections:
        class_id = int(det.get("class_id", -1))
        if class_id == 9:
            tape_color = _classify_detection_color(image, det, classifier).value
            det["tape_color"] = tape_color
            det["detected_color"] = tape_color
        elif class_id == 1:
            det["hold_color"] = HoldColor.GRAY.value
            det["detected_color"] = HoldColor.GRAY.value
        else:
            hold_color = _classify_detection_color(image, det, classifier).value
            det["hold_color"] = hold_color
            det["detected_color"] = hold_color
    return detections


def _segment_to_mask(segment: List[List[float]], image_shape: tuple[int, int, int]) -> np.ndarray | None:
    if not segment or len(segment) < 3:
        return None

    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(np.array(segment, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
    pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
    pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _center_and_diag(xyxy: List[float]) -> tuple[float, float, float]:
    x1, y1, x2, y2 = xyxy
    width = max(0.0, float(x2) - float(x1))
    height = max(0.0, float(y2) - float(y1))
    center_x = (float(x1) + float(x2)) / 2.0
    center_y = (float(y1) + float(y2)) / 2.0
    diag = math.hypot(width / 2.0, height / 2.0)
    return center_x, center_y, diag


def _classify_detection_color(
    image: np.ndarray,
    detection: Dict[str, Any],
    classifier: LabColorClassifier,
) -> HoldColor:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in detection["xyxy"]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return HoldColor.UNKNOWN

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return HoldColor.UNKNOWN

    local_mask = None
    segment_mask = _segment_to_mask(detection.get("segment", []), image.shape)
    if segment_mask is not None:
        local_mask = segment_mask[y1:y2, x1:x2].astype(bool)
        if not np.any(local_mask):
            local_mask = None

    return classifier.classify_bgr(crop, mask=local_mask)


def _parse_click_point(payload: Dict[str, Any]) -> tuple[float, float]:
    click = payload.get("click")
    if isinstance(click, dict):
        click_x = click.get("x")
        click_y = click.get("y")
    else:
        click_x = payload.get("click_x")
        click_y = payload.get("click_y")

    if click_x is None or click_y is None:
        raise ValueError("Missing click position. Provide click={x,y} or click_x/click_y.")

    try:
        return float(click_x), float(click_y)
    except (TypeError, ValueError) as exc:
        raise ValueError("click position must be numeric") from exc


def _serialize_route_hold(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "detection_id": int(item["detection_id"]),
        "class_id": int(item["class_id"]),
        "class_name": item["class_name"],
        "confidence": float(item["confidence"]),
        "xyxy": [float(v) for v in item["xyxy"]],
        "segment": item.get("segment", []),
        "hold_color": item["hold_color"],
    }


def _serialize_route_tape(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "detection_id": int(item["detection_id"]),
        "class_id": int(item["class_id"]),
        "class_name": item["class_name"],
        "confidence": float(item["confidence"]),
        "xyxy": [float(v) for v in item["xyxy"]],
        "segment": item.get("segment", []),
        "tape_color": item["tape_color"],
    }


def _build_route_from_click(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    click_point: tuple[float, float],
) -> Dict[str, Any] | None:
    holds: List[Dict[str, Any]] = []
    tapes: List[Dict[str, Any]] = []

    for det in detections:
        class_id = int(det["class_id"])
        cx, cy, diag = _center_and_diag(det["xyxy"])
        if class_id == 9:
            tape_color = str(det.get("tape_color") or det.get("detected_color") or HoldColor.UNKNOWN.value)
            tapes.append(
                {
                    "detection_id": int(det.get("id", -1)),
                    "class_id": class_id,
                    "class_name": det["class_name"],
                    "confidence": float(det["confidence"]),
                    "xyxy": [float(v) for v in det["xyxy"]],
                    "segment": det.get("segment", []),
                    "tape_color": tape_color,
                    "center_x": cx,
                    "center_y": cy,
                    "diag": diag,
                }
            )
            continue

        if class_id == 1:
            continue

        hold_color = str(det.get("hold_color") or det.get("detected_color") or HoldColor.UNKNOWN.value)
        holds.append(
            {
                "detection_id": int(det.get("id", -1)),
                "class_id": class_id,
                "class_name": det["class_name"],
                "confidence": float(det["confidence"]),
                "xyxy": [float(v) for v in det["xyxy"]],
                "segment": det.get("segment", []),
                "hold_color": hold_color,
                "center_x": cx,
                "center_y": cy,
                "diag": diag,
            }
        )

    if not holds:
        return None

    click_x, click_y = click_point
    selected_hold = min(
        holds,
        key=lambda hold: (hold["center_x"] - click_x) ** 2 + (hold["center_y"] - click_y) ** 2,
    )

    hold_color = selected_hold["hold_color"]
    route_holds = [hold for hold in holds if hold["hold_color"] == hold_color]

    hold_type_counts: Dict[str, int] = {}
    for hold in route_holds:
        class_name = hold["class_name"]
        hold_type_counts[class_name] = hold_type_counts.get(class_name, 0) + 1

    tape_color = ""
    selected_tapes: List[Dict[str, Any]] = []
    if tapes:
        adjacent_tapes_by_color: Dict[str, List[Dict[str, Any]]] = {}
        seen_tape_ids = set()

        for hold in route_holds:
            for tape in tapes:
                dist = math.hypot(
                    hold["center_x"] - tape["center_x"],
                    hold["center_y"] - tape["center_y"],
                )
                if dist >= hold["diag"] + tape["diag"]:
                    continue

                tape_id = tape["detection_id"]
                if tape_id in seen_tape_ids:
                    continue

                seen_tape_ids.add(tape_id)
                color_name = tape["tape_color"]
                if color_name == HoldColor.UNKNOWN.value:
                    continue
                adjacent_tapes_by_color.setdefault(color_name, []).append(tape)

        candidate_groups = [
            (color_name, tape_group)
            for color_name, tape_group in adjacent_tapes_by_color.items()
            if len(tape_group) >= 2
        ]

        if candidate_groups:
            tape_color, tape_group = max(
                candidate_groups,
                key=lambda item: max(t["center_y"] for t in item[1]) - min(t["center_y"] for t in item[1]),
            )
            sorted_tapes = sorted(tape_group, key=lambda tape: tape["center_y"])
            selected_tapes = [sorted_tapes[-1], sorted_tapes[0]]

    return {
        "click": {"x": click_x, "y": click_y},
        "selected_hold": _serialize_route_hold(selected_hold),
        "hold_color": hold_color,
        "tape_color": tape_color,
        "holds": [_serialize_route_hold(hold) for hold in route_holds],
        "tapes": [_serialize_route_tape(tape) for tape in selected_tapes],
        "hold_type_counts": hold_type_counts,
    }


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
        detections = _annotate_detection_colors(image, _run_inference_from_image(image, conf))

        if action in {"find_route", "route"}:
            click_point = _parse_click_point(payload)
            route = _build_route_from_click(image, detections, click_point)
            return _response(
                200,
                {
                    "ok": True,
                    "count": len(detections),
                    "detections": detections,
                    "route": route,
                },
            )

        return _response(
            200,
            {
                "ok": True,
                "count": len(detections),
                "detections": detections,
                "image_size": {
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                },
            },
        )

    except ValueError as exc:
        return _response(400, {"ok": False, "error": str(exc)})
    except Exception as exc:
        return _response(500, {"ok": False, "error": str(exc)})
