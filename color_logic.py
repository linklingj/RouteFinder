import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Tuple

import cv2
import numpy as np


DEFAULT_HSV_CONFIG_PATH = "hsv_ranges.json"


class HoldColor(str, Enum):
    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    LIGHT_BLUE = "light_blue"
    NAVY = "navy"
    PURPLE = "purple"
    PINK = "pink"
    WHITE = "white"
    GRAY = "gray"
    BLACK = "black"
    BROWN = "brown"
    UNKNOWN = "unknown"


CHROMATIC_COLORS = {
    HoldColor.RED,
    HoldColor.ORANGE,
    HoldColor.YELLOW,
    HoldColor.GREEN,
    HoldColor.BLUE,
    HoldColor.LIGHT_BLUE,
    HoldColor.NAVY,
    HoldColor.PURPLE,
    HoldColor.PINK,
    HoldColor.BROWN,
}


@dataclass(frozen=True)
class HSVRange:
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int

    def clamp(self) -> "HSVRange":
        return HSVRange(
            h_min=max(0, min(179, int(self.h_min))),
            h_max=max(0, min(179, int(self.h_max))),
            s_min=max(0, min(255, int(self.s_min))),
            s_max=max(0, min(255, int(self.s_max))),
            v_min=max(0, min(255, int(self.v_min))),
            v_max=max(0, min(255, int(self.v_max))),
        )

    def to_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        r = self.clamp()
        if r.h_min <= r.h_max:
            lower = np.array([r.h_min, r.s_min, r.v_min], dtype=np.uint8)
            upper = np.array([r.h_max, r.s_max, r.v_max], dtype=np.uint8)
            return cv2.inRange(hsv_image, lower, upper)

        lower_1 = np.array([0, r.s_min, r.v_min], dtype=np.uint8)
        upper_1 = np.array([r.h_max, r.s_max, r.v_max], dtype=np.uint8)
        lower_2 = np.array([r.h_min, r.s_min, r.v_min], dtype=np.uint8)
        upper_2 = np.array([179, r.s_max, r.v_max], dtype=np.uint8)
        mask_1 = cv2.inRange(hsv_image, lower_1, upper_1)
        mask_2 = cv2.inRange(hsv_image, lower_2, upper_2)
        return cv2.bitwise_or(mask_1, mask_2)


DEFAULT_HSV_RANGES: Dict[HoldColor, List[HSVRange]] = {
    HoldColor.RED: [
        HSVRange(0, 10, 70, 255, 40, 255),
        HSVRange(170, 179, 70, 255, 40, 255),
    ],
    HoldColor.ORANGE: [HSVRange(11, 22, 70, 255, 60, 255)],
    HoldColor.YELLOW: [HSVRange(23, 35, 60, 255, 80, 255)],
    HoldColor.GREEN: [HSVRange(36, 85, 50, 255, 40, 255)],
    HoldColor.BLUE: [HSVRange(86, 120, 60, 255, 40, 255)],
    HoldColor.LIGHT_BLUE: [HSVRange(100, 130, 50, 255, 40, 255)],
    HoldColor.NAVY: [HSVRange(100, 130, 80, 255, 20, 150)],
    HoldColor.PURPLE: [HSVRange(121, 150, 50, 255, 40, 255)],
    HoldColor.PINK: [HSVRange(151, 169, 40, 255, 80, 255)],
    HoldColor.BROWN: [HSVRange(8, 25, 80, 255, 20, 180)],
    HoldColor.WHITE: [HSVRange(0, 179, 0, 45, 180, 255)],
    HoldColor.GRAY: [HSVRange(0, 179, 0, 45, 60, 179)],
    HoldColor.BLACK: [HSVRange(0, 179, 0, 255, 0, 59)],
}


def _copy_ranges(ranges: Mapping[HoldColor, List[HSVRange]]) -> Dict[HoldColor, List[HSVRange]]:
    return {color: [HSVRange(**asdict(r)) for r in entries] for color, entries in ranges.items()}


def parse_hold_color(value: str) -> HoldColor:
    text = value.strip().lower()
    for color in HoldColor:
        if color.value == text:
            return color
    raise ValueError(f"Unsupported color name: {value}")


def color_to_bgr(color: HoldColor) -> Tuple[int, int, int]:
    palette = {
        HoldColor.RED: (0, 0, 255),
        HoldColor.ORANGE: (0, 165, 255),
        HoldColor.YELLOW: (0, 255, 255),
        HoldColor.GREEN: (0, 200, 0),
        HoldColor.BLUE: (255, 120, 0),
        HoldColor.LIGHT_BLUE: (255, 200, 0),
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


def load_hsv_ranges(config_path: str = DEFAULT_HSV_CONFIG_PATH) -> Dict[HoldColor, List[HSVRange]]:
    path = Path(config_path)
    if not path.exists():
        return _copy_ranges(DEFAULT_HSV_RANGES)

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    loaded = _copy_ranges(DEFAULT_HSV_RANGES)
    for key, ranges in payload.items():
        try:
            color = parse_hold_color(key)
        except ValueError:
            continue
        loaded[color] = [HSVRange(**entry).clamp() for entry in ranges]
    return loaded


def save_hsv_ranges(
    ranges: Mapping[HoldColor, List[HSVRange]],
    config_path: str = DEFAULT_HSV_CONFIG_PATH,
) -> None:
    serializable: MutableMapping[str, List[Dict[str, int]]] = {}
    for color, entries in ranges.items():
        serializable[color.value] = [asdict(entry.clamp()) for entry in entries]

    with Path(config_path).open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


class HSVColorClassifier:
    def __init__(
        self,
        ranges: Mapping[HoldColor, List[HSVRange]] | None = None,
        *,
        min_match_ratio: float = 0.03,
        chromatic_preference_ratio: float = 0.42,
    ):
        source = ranges if ranges is not None else DEFAULT_HSV_RANGES
        self.ranges = _copy_ranges(source)
        self.min_match_ratio = min_match_ratio
        self.chromatic_preference_ratio = chromatic_preference_ratio

    @classmethod
    def from_config(cls, config_path: str = DEFAULT_HSV_CONFIG_PATH) -> "HSVColorClassifier":
        return cls(load_hsv_ranges(config_path))

    def score_hsv(self, hsv_image: np.ndarray) -> Dict[HoldColor, int]:
        scores: Dict[HoldColor, int] = {}
        for color, entries in self.ranges.items():
            if not entries:
                scores[color] = 0
                continue
            mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            for entry in entries:
                mask = cv2.bitwise_or(mask, entry.to_mask(hsv_image))
            scores[color] = int(cv2.countNonZero(mask))
        return scores

    def classify_hsv(self, hsv_image: np.ndarray) -> HoldColor:
        total_pixels = int(hsv_image.shape[0] * hsv_image.shape[1])
        if total_pixels <= 0:
            return HoldColor.UNKNOWN

        scores = self.score_hsv(hsv_image)
        best_color, best_count = max(scores.items(), key=lambda item: item[1], default=(HoldColor.UNKNOWN, 0))
        best_ratio = best_count / total_pixels
        if best_ratio < self.min_match_ratio:
            return HoldColor.UNKNOWN

        chromatic = [(color, count) for color, count in scores.items() if color in CHROMATIC_COLORS]
        if chromatic:
            chroma_color, chroma_count = max(chromatic, key=lambda item: item[1])
            chroma_ratio = chroma_count / total_pixels
            if chroma_ratio >= self.chromatic_preference_ratio:
                return chroma_color

        return best_color

    def classify_bgr(self, bgr_image: np.ndarray) -> HoldColor:
        if bgr_image.size == 0:
            return HoldColor.UNKNOWN
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        return self.classify_hsv(hsv_image)


def tune_hsv_range(
    image_bgr: np.ndarray,
    classifier: HSVColorClassifier,
    *,
    color: HoldColor,
    range_index: int = 0,
    config_path: str = DEFAULT_HSV_CONFIG_PATH,
) -> None:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Image is empty.")
    if range_index < 0:
        raise ValueError("range_index must be >= 0")

    ranges = classifier.ranges.setdefault(color, [])
    while len(ranges) <= range_index:
        ranges.append(HSVRange(0, 179, 0, 255, 0, 255))

    base = ranges[range_index].clamp()
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    window_name = f"HSV Tuner - {color.value}[{range_index}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("H min", window_name, base.h_min, 179, lambda _: None)
    cv2.createTrackbar("H max", window_name, base.h_max, 179, lambda _: None)
    cv2.createTrackbar("S min", window_name, base.s_min, 255, lambda _: None)
    cv2.createTrackbar("S max", window_name, base.s_max, 255, lambda _: None)
    cv2.createTrackbar("V min", window_name, base.v_min, 255, lambda _: None)
    cv2.createTrackbar("V max", window_name, base.v_max, 255, lambda _: None)

    current = base
    print("HSV tuner controls: press 's' to save, 'q' or ESC to exit.")

    while True:
        current = HSVRange(
            h_min=cv2.getTrackbarPos("H min", window_name),
            h_max=cv2.getTrackbarPos("H max", window_name),
            s_min=cv2.getTrackbarPos("S min", window_name),
            s_max=cv2.getTrackbarPos("S max", window_name),
            v_min=cv2.getTrackbarPos("V min", window_name),
            v_max=cv2.getTrackbarPos("V max", window_name),
        ).clamp()

        mask = current.to_mask(hsv_image)
        masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        preview = np.hstack([image_bgr, mask_bgr, masked])

        cv2.putText(
            preview,
            f"{color.value}[{range_index}] - s:save q:quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            ranges[range_index] = current
            save_hsv_ranges(classifier.ranges, config_path=config_path)
            print(f"Saved {color.value}[{range_index}] to {config_path}: {asdict(current)}")

    ranges[range_index] = current
    cv2.destroyWindow(window_name)
