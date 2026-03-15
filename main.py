import argparse
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from color_logic import (
    DEFAULT_HSV_CONFIG_PATH,
    HSVColorClassifier,
    HoldColor,
    color_to_bgr,
    parse_hold_color,
    tune_hsv_range,
)
from model_loader import get_model

DEFAULT_OVERLAY_ALPHA = 0.5


class RouteDifficulty(str, Enum):
    VB = "VB"
    V0 = "V0"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"
    V6 = "V6"
    V7 = "V7"
    V8 = "V8+"


@dataclass
class Hold:
    hold_type: int
    hold_type_str: str
    confidence: float
    xyxy: List[float]
    segment: List[List[float]] = field(default_factory=list)
    hold_color: HoldColor = HoldColor.UNKNOWN


@dataclass
class Tape:
    confidence: float
    xyxy: List[float]
    segment: List[List[float]] = field(default_factory=list)
    tape_color: HoldColor = HoldColor.UNKNOWN


class ImageInfo:
    img: cv2.Mat
    holds: List[Hold]
    down_holds: List[Hold]
    tapes: List[Tape]

    def __init__(self, img: cv2.Mat, detections: List[Dict], color_classifier: HSVColorClassifier):
        self.img = img
        self.holds = []
        self.down_holds = []
        self.tapes = []
        self.color_classifier = color_classifier

        for det in detections:
            xyxy = [float(v) for v in det["xyxy"]]
            segment = [[float(p[0]), float(p[1])] for p in det.get("segment", [])]

            if det["class_id"] == 9:
                self.tapes.append(
                    Tape(
                        confidence=float(det["confidence"]),
                        xyxy=xyxy,
                        segment=segment,
                        tape_color=self.get_color(xyxy),
                    )
                )
            elif det["class_id"] == 1:
                self.down_holds.append(
                    Hold(
                        hold_type=int(det["class_id"]),
                        hold_type_str=det["class_name"],
                        confidence=float(det["confidence"]),
                        xyxy=xyxy,
                        segment=segment,
                        hold_color=self.get_color(xyxy),
                    )
                )
            else:
                self.holds.append(
                    Hold(
                        hold_type=int(det["class_id"]),
                        hold_type_str=det["class_name"],
                        confidence=float(det["confidence"]),
                        xyxy=xyxy,
                        segment=segment,
                        hold_color=self.get_color(xyxy),
                    )
                )

    def get_color(self, xyxy: List[float]) -> HoldColor:
        x1, y1, x2, y2 = map(int, xyxy)
        crop = self.img[y1:y2, x1:x2]
        return self.color_classifier.classify_bgr(crop)


@dataclass
class Route:
    gym: str
    difficulty: RouteDifficulty
    hold_color: str = ""
    tape_color: str = ""
    holds: List[Hold] = field(default_factory=list)
    start_tape: Tape = None
    end_tape: Tape = None

    def add_hold(self, hold: Hold) -> None:
        self.holds.append(hold)

    def set_route(self, img_info: ImageInfo, ref_hold_pos) -> bool:
        if not img_info.holds:
            return False

        self.holds.clear()
        self.start_tape = None
        self.end_tape = None
        self.hold_color = ""
        self.tape_color = ""

        min_hold_dist, closest_hold = float("inf"), None
        for hold in img_info.holds:
            hold_center = (
                (hold.xyxy[0] + hold.xyxy[2]) / 2,
                (hold.xyxy[1] + hold.xyxy[3]) / 2,
            )
            dist = ((hold_center[0] - ref_hold_pos[0]) ** 2 + (hold_center[1] - ref_hold_pos[1]) ** 2) ** 0.5
            if dist < min_hold_dist:
                min_hold_dist, closest_hold = dist, hold

        if not closest_hold:
            return False

        self.visualize_hsv(img_info, closest_hold)

        self.hold_color = closest_hold.hold_color.value
        for hold in img_info.holds:
            if hold.hold_color == closest_hold.hold_color:
                self.add_hold(hold)

        if img_info.tapes:
            min_tape_dist, closest_tape = float("inf"), None
            for tape in img_info.tapes:
                tape_center = (
                    (tape.xyxy[0] + tape.xyxy[2]) / 2,
                    (tape.xyxy[1] + tape.xyxy[3]) / 2,
                )
                dist = ((tape_center[0] - ref_hold_pos[0]) ** 2 + (tape_center[1] - ref_hold_pos[1]) ** 2) ** 0.5
                if dist < min_tape_dist:
                    min_tape_dist, closest_tape = dist, tape
            if closest_tape:
                self.start_tape = closest_tape
                self.tape_color = closest_tape.tape_color.value

        print(
            f"Route set: hold_color={self.hold_color}, tape_color={self.tape_color}, "
            f"holds={len(self.holds)}, start_tape={self.start_tape is not None}"
        )
        return True

    def to_dict(self) -> Dict:
        return {
            "gym": self.gym,
            "difficulty": self.difficulty.value,
            "holds": [
                {
                    "hold_type": hold.hold_type,
                    "confidence": hold.confidence,
                    "xyxy": hold.xyxy,
                }
                for hold in self.holds
            ],
        }

    def visualize_hsv(self, img_info: ImageInfo, closest_hold: Hold) -> None:
        hsv = []
        hsv_img = cv2.cvtColor(img_info.img, cv2.COLOR_BGR2HSV)
        for point in closest_hold.segment:
            point_hsv = hsv_img[int(point[1]), int(point[0])]
            hsv.append(point_hsv)

        if not hsv:
            return

        plt.figure(figsize=(6, 4))
        ax = plt.subplot(111, projection="3d")
        ax.set_xlabel("Hue (OpenCV)")
        ax.set_ylabel("Saturation")
        ax.set_zlabel("Value")
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)
        ax.scatter([h[0] for h in hsv], [h[1] for h in hsv], [h[2] for h in hsv], c="blue", alpha=0.5)
        plt.title("HSV Distribution")
        plt.show()


def _draw_objects(
    base_img: cv2.Mat,
    holds: List[Hold],
    tapes: List[Tape],
    overlay_alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> cv2.Mat:
    canvas = base_img.copy()
    overlay = canvas.copy()
    overlay_alpha = max(0.0, min(1.0, float(overlay_alpha)))

    for hold in holds:
        color = color_to_bgr(hold.hold_color)
        if hold.segment:
            pts = np.array(hold.segment, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        x1, y1, x2, y2 = map(int, hold.xyxy)
        #cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        cv2.putText(canvas, hold.hold_type_str, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for tape in tapes:
        color = color_to_bgr(tape.tape_color)
        if tape.segment:
            pts = np.array(tape.segment, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        x1, y1, x2, y2 = map(int, tape.xyxy)
        cv2.putText(canvas, f"tape:{tape.tape_color.value}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.addWeighted(overlay, overlay_alpha, canvas, 1.0 - overlay_alpha, 0, canvas)
    return canvas


def _extract_detections(image_path: str) -> Tuple[cv2.Mat, List[Dict]]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    results = get_model().predict(source=image_path, device="cpu", verbose=False)
    detections: List[Dict] = []
    if not results:
        return image, detections

    result = results[0]
    masks_xy = result.masks.xy if result.masks is not None else []
    if result.boxes is None:
        return image, detections

    for i, box in enumerate(result.boxes):
        segment = []
        if i < len(masks_xy):
            segment = masks_xy[i].tolist()
        detections.append(
            {
                "class_id": int(box.cls[0]),
                "class_name": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "xyxy": [float(v) for v in box.xyxy[0].tolist()],
                "segment": segment,
            }
        )
    return image, detections


def _on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    img_info: ImageInfo = param["img_info"]
    route = Route(gym="unknown", difficulty=RouteDifficulty.V0)

    if not route.set_route(img_info, (x, y)):
        print("Could not build route from the clicked position.")
        return

    overlay_alpha = param.get("overlay_alpha", DEFAULT_OVERLAY_ALPHA)
    selected_tapes = [route.start_tape] if route.start_tape else []
    route_image = _draw_objects(img_info.img, route.holds, selected_tapes, overlay_alpha=overlay_alpha)
    cv2.imshow("Route Segmentation", route_image)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RouteFinder")
    parser.add_argument("--image", dest="image_path", help="Input image path")
    parser.add_argument("--hsv-config", default=DEFAULT_HSV_CONFIG_PATH, help="HSV range config JSON path")
    parser.add_argument("--tune-hsv", action="store_true", help="Run HSV range tuning UI instead of detection")
    parser.add_argument("--tune-color", default=HoldColor.RED.value, help="Color name to tune (e.g. red, blue)")
    parser.add_argument("--tune-range-index", type=int, default=0, help="Range index for selected color")
    parser.add_argument("--overlay-alpha", type=float, default=DEFAULT_OVERLAY_ALPHA, help="Overlay alpha (0.0~1.0)")
    return parser.parse_args()


def main():
    args = _parse_args()
    image_path = args.image_path or input("Enter image path: ").strip()
    if not image_path:
        print("Image path is empty.")
        return
    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
        return

    classifier = HSVColorClassifier.from_config(args.hsv_config)
    overlay_alpha = max(0.0, min(1.0, float(args.overlay_alpha)))

    if args.tune_hsv:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot load image: {image_path}")
            return
        try:
            tune_color = parse_hold_color(args.tune_color)
        except ValueError as e:
            print(str(e))
            return

        tune_hsv_range(
            image,
            classifier,
            color=tune_color,
            range_index=args.tune_range_index,
            config_path=args.hsv_config,
        )
        return

    image, detections = _extract_detections(image_path)
    img_info = ImageInfo(image, detections, color_classifier=classifier)
    segmented = _draw_objects(img_info.img, img_info.holds, img_info.tapes, overlay_alpha=overlay_alpha)

    window_name = "Segmented Holds (click to build route)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _on_click, {"img_info": img_info, "overlay_alpha": 0.9})
    cv2.imshow(window_name, segmented)
    print("Click on the image to build a route from the nearest hold. Press any key to exit.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
