from ultralytics import YOLO

MODEL_PATH = "models/0314-2.pt"
_model = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model
