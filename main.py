import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import cv2
import numpy as np

from model_loader import get_model
from infer import _run_inference


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


class HoldColor(str, Enum):
    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    NAVY = "navy"
    PURPLE = "purple"
    PINK = "pink"
    WHITE = "white"
    GRAY = "gray"
    BLACK = "black"
    BROWN = "brown"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HueRange:
    color: HoldColor
    min_deg: int
    max_deg: int

    def contains(self, hue_deg: int) -> bool:
        if self.min_deg <= self.max_deg:
            return self.min_deg <= hue_deg <= self.max_deg
        return hue_deg >= self.min_deg or hue_deg <= self.max_deg


DEFAULT_HUE_RANGES: Tuple[HueRange, ...] = (
    HueRange(HoldColor.RED, 345, 15),
    HueRange(HoldColor.ORANGE, 16, 40),
    HueRange(HoldColor.YELLOW, 41, 70),
    HueRange(HoldColor.GREEN, 71, 165),
    HueRange(HoldColor.BLUE, 166, 255),
    HueRange(HoldColor.PURPLE, 256, 290),
    HueRange(HoldColor.PINK, 291, 344),
)


def hue_to_hold_color(
    hue: int,
    *,
    saturation: int = 255,
    value: int = 255,
    hue_scale: str = "opencv",
    hue_ranges: Tuple[HueRange, ...] = DEFAULT_HUE_RANGES,
) -> HoldColor:
    # HSV 기반 무채색 우선 분기
    if value <= 50:
        return HoldColor.BLACK
    if saturation <= 35:
        if value >= 200:
            return HoldColor.WHITE
        return HoldColor.GRAY

    if hue_scale == "opencv":
        hue_deg = int(round((hue % 180) * 2))
    elif hue_scale == "degree":
        hue_deg = hue % 360
    else:
        raise ValueError("hue_scale must be either 'opencv' or 'degree'")

    for band in hue_ranges:
        if band.contains(hue_deg):
            return band.color
    return HoldColor.UNKNOWN


@dataclass
class Hold:
    hold_type: int
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
    tapes: List[Tape]

    def __init__(self, img: cv2.Mat, detections: List[Dict]):
        self.img = img
        self.holds = []
        self.tapes = []
        for det in detections:
            xyxy = [float(v) for v in det["xyxy"]]
            segment = [[float(p[0]), float(p[1])] for p in det.get("segment", [])]
            if det["class_id"] == 10:
                self.tapes.append(
                    Tape(
                        confidence=float(det["confidence"]),
                        xyxy=xyxy,
                        segment=segment,
                        tape_color=self.get_color(xyxy),
                    )
                )
            else:
                self.holds.append(
                    Hold(
                        hold_type=int(det["class_id"]),
                        confidence=float(det["confidence"]),
                        xyxy=xyxy,
                        segment=segment,
                        hold_color=self.get_color(xyxy),
                    )
                )

    def get_color(self, xyxy: List[float]) -> HoldColor:
        x1, y1, x2, y2 = map(int, xyxy)
        crop = self.img[y1:y2, x1:x2]
        if crop.size == 0:
            return HoldColor.UNKNOWN
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue_channel = hsv_crop[:, :, 0]
        sat_channel = hsv_crop[:, :, 1]
        val_channel = hsv_crop[:, :, 2]
        mean_hue = int(hue_channel.mean())
        mean_sat = int(sat_channel.mean())
        mean_val = int(val_channel.mean())
        return hue_to_hold_color(mean_hue, saturation=mean_sat, value=mean_val)


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


def _color_bgr(color: HoldColor) -> Tuple[int, int, int]:
    palette = {
        HoldColor.RED: (0, 0, 255),
        HoldColor.ORANGE: (0, 165, 255),
        HoldColor.YELLOW: (0, 255, 255),
        HoldColor.GREEN: (0, 200, 0),
        HoldColor.BLUE: (255, 120, 0),
        HoldColor.NAVY: (180, 60, 0),
        HoldColor.PURPLE: (180, 0, 180),
        HoldColor.PINK: (203, 192, 255),
        HoldColor.WHITE: (255, 255, 255),
        HoldColor.GRAY: (150, 150, 150),
        HoldColor.BLACK: (30, 30, 30),
        HoldColor.BROWN: (42, 42, 165),
        HoldColor.UNKNOWN: (128, 128, 128),
    }
    return palette.get(color, (128, 128, 128))


def _draw_objects(base_img: cv2.Mat, holds: List[Hold], tapes: List[Tape]) -> cv2.Mat:
    canvas = base_img.copy()
    overlay = canvas.copy()

    for hold in holds:
        color = _color_bgr(hold.hold_color)
        if hold.segment:
            pts = np.array(hold.segment, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        x1, y1, x2, y2 = map(int, hold.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, hold.hold_color.value, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for tape in tapes:
        color = _color_bgr(tape.tape_color)
        if tape.segment:
            pts = np.array(tape.segment, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        x1, y1, x2, y2 = map(int, tape.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"tape:{tape.tape_color.value}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0, canvas)
    return canvas


def _extract_detections(image_path: str) -> Tuple[cv2.Mat, List[Dict]]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

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
        print("선택한 위치에서 route를 구성하지 못했습니다.")
        return

    selected_tapes = [route.start_tape] if route.start_tape else []
    route_image = _draw_objects(img_info.img, route.holds, selected_tapes)
    cv2.imshow("Route Segmentation", route_image)


def main():
    image_path = input("이미지 경로를 입력하세요: ").strip()
    if not image_path:
        print("이미지 경로가 비어 있습니다.")
        return
    if not os.path.exists(image_path):
        print(f"파일이 존재하지 않습니다: {image_path}")
        return

    image, detections = _extract_detections(image_path)
    img_info = ImageInfo(image, detections)

    segmented = _draw_objects(img_info.img, img_info.holds, img_info.tapes)

    window_name = "Segmented Holds (click to build route)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _on_click, {"img_info": img_info})
    cv2.imshow(window_name, segmented)
    print("이미지 위를 클릭하면 클릭 지점 기준 route 홀드만 별도 창에 표시됩니다. 종료: 아무 키")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
