import argparse
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from matplotlib import pyplot as plt

from color_logic import (
    DEFAULT_LAB_CONFIG_PATH,
    LabColorClassifier,
    HoldColor,
    apply_retinex,
    color_to_bgr,
    parse_hold_color,
    tune_lab_range,
)
from model_loader import get_model
from infer import run_inference

DEFAULT_OVERLAY_ALPHA = 0.1


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
    pixels: List[Tuple[int, int]] = field(default_factory=list)
    hold_color: HoldColor = HoldColor.UNKNOWN
    color_ratios: Dict[HoldColor, float] = field(default_factory=dict)
    hold_crop: np.ndarray | None = None


@dataclass
class Tape:
    confidence: float
    xyxy: List[float]
    segment: List[List[float]] = field(default_factory=list)
    pixels: List[Tuple[int, int]] = field(default_factory=list)
    tape_color: HoldColor = HoldColor.UNKNOWN


class ImageInfo:
    img: cv2.Mat
    retinex_img: cv2.Mat
    lab_img: cv2.Mat
    holds: List[Hold]
    down_holds: List[Hold]
    tapes: List[Tape]

    def __init__(self, img: cv2.Mat, detections: List[Dict], color_classifier: LabColorClassifier):
        self.img = img
        # Keep preprocessing identical to LAB tuner: Retinex on full image, then LAB conversion.
        self.retinex_img = apply_retinex(self.img)
        self.lab_img = cv2.cvtColor(self.retinex_img, cv2.COLOR_BGR2LAB)
        self.holds = []
        self.down_holds = []
        self.tapes = []
        self.color_classifier = color_classifier

        for det in detections:
            xyxy = [float(v) for v in det["xyxy"]]
            segment = [[float(p[0]), float(p[1])] for p in det.get("segment", [])]
            pixels = [(x, y) for y, row in enumerate(det.get("pixels", [])) for x, val in enumerate(row) if val]

            if det["class_id"] == 9:
                self.tapes.append(
                    Tape(
                        confidence=float(det["confidence"]),
                        xyxy=xyxy,
                        segment=segment,
                        pixels=pixels,
                        tape_color=self.get_color(xyxy, pixels),
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
                        pixels=pixels,
                        hold_color=HoldColor.GRAY,
                        color_ratios={HoldColor.GRAY: 1.0},
                    )
                )
            else:
                hold_color, color_ratios, hold_crop = self.get_color_and_ratios(xyxy, pixels)
                self.holds.append(
                    Hold(
                        hold_type=int(det["class_id"]),
                        hold_type_str=det["class_name"],
                        confidence=float(det["confidence"]),
                        xyxy=xyxy,
                        segment=segment,
                        pixels=pixels,
                        hold_color=hold_color,
                        color_ratios=color_ratios,
                        hold_crop=hold_crop,
                    )
                )

    def get_color(self, xyxy: List[float], pixels: List[Tuple[int, int]]) -> HoldColor:
        color, _, _ = self.get_color_and_ratios(xyxy, pixels)
        return color

    def get_color_and_ratios(
        self,
        xyxy: List[float],
        pixels: List[Tuple[int, int]],
    ) -> Tuple[HoldColor, Dict[HoldColor, float], np.ndarray | None]:
        h, w = self.img.shape[:2]
        x1, y1, x2, y2 = map(int, xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return HoldColor.UNKNOWN, {}, None

        crop = self.img[y1:y2, x1:x2]
        if crop.size == 0:
            return HoldColor.UNKNOWN, {}, None

        # Keep a snapshot of the crop made during color classification for click-time preview.
        saved_crop = crop.copy()
        crop_lab = self.lab_img[y1:y2, x1:x2]

        local_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        for x, y in pixels:
            if x1 <= x < x2 and y1 <= y < y2:
                local_mask[y - y1, x - x1] = 255

        lab_for_scoring = crop_lab
        if cv2.countNonZero(local_mask) > 0:
            selected = crop_lab[local_mask.astype(bool)]
            if selected.size > 0:
                # Use only segmented hold pixels to avoid background-biased colors.
                lab_for_scoring = selected.reshape((-1, 1, 3))

        scores = self.color_classifier.score_lab(lab_for_scoring)
        total_pixels = int(lab_for_scoring.shape[0] * lab_for_scoring.shape[1])
        ratios = {
            color: (count / total_pixels) if total_pixels > 0 else 0.0
            for color, count in scores.items()
        }
        hold_color = self.color_classifier.classify_lab(lab_for_scoring)

        return hold_color, ratios, saved_crop


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

        # self.visualize_hold_crop(closest_hold)
        # self.visualize_color_ratios(closest_hold)

        self.hold_color = closest_hold.hold_color.value
        for hold in img_info.holds:
            if hold.hold_color == closest_hold.hold_color:
                self.add_hold(hold)

        # 홀드들에서 인접 테이프 모두 저장
        # 두개 이상 같은 색 테이프 있으면 스타트와 엔드로 저장

        if img_info.tapes:
            tape_meta = []
            for tape in img_info.tapes:
                tape_w = tape.xyxy[2] - tape.xyxy[0]
                tape_h = tape.xyxy[3] - tape.xyxy[1]
                tape_center = ((tape.xyxy[0] + tape.xyxy[2]) / 2, (tape.xyxy[1] + tape.xyxy[3]) / 2)
                tape_diag = ((tape_w/2) ** 2 + (tape_h/2) ** 2) ** 0.5
                tape_meta.append((tape, tape_center, tape_diag))

            adj_tapes_by_color: Dict[HoldColor, List[Tape]] = {}
            seen_tape_ids = set()

            for hold in self.holds:
                hold_w = hold.xyxy[2] - hold.xyxy[0]
                hold_h = hold.xyxy[3] - hold.xyxy[1]
                hold_center = ((hold.xyxy[0] + hold.xyxy[2]) / 2, (hold.xyxy[1] + hold.xyxy[3]) / 2)
                hold_diag = ((hold_w/2) ** 2 + (hold_h/2) ** 2) ** 0.5

                for tape, tape_center, tape_diag in tape_meta:
                    dist = ((hold_center[0] - tape_center[0]) ** 2 + (hold_center[1] - tape_center[1]) ** 2) ** 0.5
                    # 간단한 인접 조건: 중심 간 거리 < 홀드 대각선 길이 + 테이프 대각선 길이
                    if dist >= hold_diag + tape_diag:
                        continue

                    tape_id = id(tape)
                    if tape_id in seen_tape_ids:
                        continue

                    seen_tape_ids.add(tape_id)
                    if tape.tape_color not in adj_tapes_by_color:
                        adj_tapes_by_color[tape.tape_color] = []
                    adj_tapes_by_color[tape.tape_color].append(tape)

            candidate_groups = [
                (color, tapes)
                for color, tapes in adj_tapes_by_color.items()
                if color != HoldColor.UNKNOWN and len(tapes) >= 2
            ]

            if candidate_groups:
                # 가장 위/아래 간격이 큰 색을 선택
                selected_color, selected_tapes = max(
                    candidate_groups,
                    key=lambda item: (
                        max(((t.xyxy[1] + t.xyxy[3]) / 2) for t in item[1])
                        - min(((t.xyxy[1] + t.xyxy[3]) / 2) for t in item[1]),
                    ),
                )
                sorted_by_y = sorted(selected_tapes, key=lambda t: (t.xyxy[1] + t.xyxy[3]) / 2)
                self.end_tape = sorted_by_y[0]
                self.start_tape = sorted_by_y[-1]
                self.tape_color = selected_color.value

        print(
            f"Route set: hold_color={self.hold_color}, tape_color={self.tape_color}, "
            f"holds={len(self.holds)}, start_tape={self.start_tape is not None}, end_tape={self.end_tape is not None}"
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

    def visualize_color_ratios(self, closest_hold: Hold) -> None:
        if not closest_hold.color_ratios:
            return

        sorted_items = sorted(closest_hold.color_ratios.items(), key=lambda item: item[1], reverse=True)
        labels = [color.value for color, ratio in sorted_items if ratio > 0]
        values = [ratio * 100.0 for _, ratio in sorted_items if ratio > 0]

        if not labels:
            return

        bar_colors = []
        for color, ratio in sorted_items:
            if ratio <= 0:
                continue
            bgr = color_to_bgr(color)
            bar_colors.append((bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0))

        plt.figure(figsize=(8, 4))
        bars = plt.bar(labels, values, color=bar_colors)
        plt.ylim(0, 100)
        plt.ylabel("ratio (%)")
        plt.title("Clicked Hold Color Ratios")
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.show()

    def visualize_hold_crop(self, closest_hold: Hold) -> None:
        if closest_hold.hold_crop is None or closest_hold.hold_crop.size == 0:
            return
        cv2.imshow("Clicked Hold Crop", closest_hold.hold_crop)


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


def _draw_route(
    base_img: cv2.Mat, 
    holds: List[Hold],
    tapes: List[Tape],
    overlay_alpha: float = DEFAULT_OVERLAY_ALPHA
) -> cv2.Mat:
    # all gray scale except the route holds and tapes
    gray_canvas = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor(gray_canvas, cv2.COLOR_GRAY2BGR)
    
    for hold in holds:
        for (x, y) in hold.pixels:
            canvas[y, x] = base_img[y, x]
    for tape in tapes:
        for (x, y) in tape.pixels:
            canvas[y, x] = color_to_bgr(tape.tape_color)

    return canvas


def _on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    img_info: ImageInfo = param["img_info"]
    route = Route(gym="unknown", difficulty=RouteDifficulty.V0)

    if not route.set_route(img_info, (x, y)):
        print("Could not build route from the clicked position.")
        return

    overlay_alpha = param.get("overlay_alpha", DEFAULT_OVERLAY_ALPHA)
    selected_tapes = []
    selected_tapes.append(route.start_tape) if route.start_tape else None
    selected_tapes.append(route.end_tape) if route.end_tape else None
    # route_image = _draw_objects(img_info.img, route.holds, selected_tapes, overlay_alpha=overlay_alpha)
    route_image = _draw_route(img_info.img, route.holds, selected_tapes, overlay_alpha=overlay_alpha)
    cv2.imshow("Route Segmentation", route_image)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RouteFinder")
    parser.add_argument("--image", dest="image_path", help="Input image path")
    parser.add_argument("--lab-config", default=DEFAULT_LAB_CONFIG_PATH, help="LAB range config JSON path")
    parser.add_argument("--tune-lab", action="store_true", help="Run LAB range tuning UI instead of detection")
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

    classifier = LabColorClassifier.from_config(args.lab_config)
    overlay_alpha = max(0.0, min(1.0, float(args.overlay_alpha)))

    if args.tune_lab:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot load image: {image_path}")
            return
        try:
            tune_color = parse_hold_color(args.tune_color)
        except ValueError as e:
            print(str(e))
            return

        tune_lab_range(
            image,
            classifier,
            color=tune_color,
            range_index=args.tune_range_index,
            config_path=args.lab_config,
        )
        return

    image, detections = run_inference(image_path)
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