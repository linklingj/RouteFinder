import json
from typing import Any, Dict

import infer
from model_loader import MODEL_PATH


def _response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        action = event.get("action", "predict")

        if action == "health":
            return _response(
                200,
                {
                    "ok": True,
                    "message": "Lambda is healthy",
                    "model_path": MODEL_PATH,
                },
            )

        image_base64 = event.get("image_base64")
        if not image_base64:
            return _response(
                400,
                {
                    "ok": False,
                    "error": "Missing `image_base64` in request event",
                },
            )

        detections = infer._run_inference(image_base64)
        return _response(200, {"ok": True, "detections": detections})

    except Exception as exc:
        return _response(500, {"ok": False, "error": str(exc)})
