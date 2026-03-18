import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt

def single_scale_retinex(img: np.ndarray, sigma: float) -> np.ndarray:
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex


def multi_scale_retinex(img: np.ndarray, sigma_list: List[float]) -> np.ndarray:
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += single_scale_retinex(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex


def color_restoration(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration


def apply_retinex(img: np.ndarray) -> np.ndarray:
    if img.size == 0:
        return img
    sigma_list = [15.0, 80.0, 250.0]
    G = 5.0
    b = 25.0
    alpha = 125.0
    beta = 46.0

    img_float = np.float64(img) + 1.0
    msr = multi_scale_retinex(img_float, sigma_list)
    cr = color_restoration(img_float, alpha, beta)
    msrcr = G * (msr * cr - b)

    # Normalize each channel robustly so the image does not collapse to near-black.
    normalized = np.zeros_like(msrcr)
    for ch in range(msrcr.shape[2]):
        channel = msrcr[:, :, ch]
        low, high = np.percentile(channel, (1, 99))
        if high - low < 1e-6:
            normalized[:, :, ch] = np.clip(channel, 0, 255)
            continue
        normalized[:, :, ch] = (channel - low) * (255.0 / (high - low))

    normalized = np.clip(normalized, 0, 255)
    return np.uint8(normalized)


DEFAULT_LAB_CONFIG_PATH = "lab_ranges.json"


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
class LabRange:
    l_min: int
    l_max: int
    a_min: int
    a_max: int
    b_min: int
    b_max: int

    def clamp(self) -> "LabRange":
        return LabRange(
            l_min=max(0, min(255, int(self.l_min))),
            l_max=max(0, min(255, int(self.l_max))),
            a_min=max(0, min(255, int(self.a_min))),
            a_max=max(0, min(255, int(self.a_max))),
            b_min=max(0, min(255, int(self.b_min))),
            b_max=max(0, min(255, int(self.b_max))),
        )

    def to_mask(self, lab_image: np.ndarray) -> np.ndarray:
        r = self.clamp()
        lower = np.array([r.l_min, r.a_min, r.b_min], dtype=np.uint8)
        upper = np.array([r.l_max, r.a_max, r.b_max], dtype=np.uint8)
        return cv2.inRange(lab_image, lower, upper)


DEFAULT_LAB_RANGES: Dict[HoldColor, List[LabRange]] = {
    HoldColor.RED: [LabRange(0, 255, 150, 255, 0, 255)],
    HoldColor.ORANGE: [LabRange(0, 255, 140, 180, 150, 255)],
    HoldColor.YELLOW: [LabRange(0, 255, 110, 140, 150, 255)],
    HoldColor.GREEN: [LabRange(0, 255, 0, 120, 128, 255)],
    HoldColor.BLUE: [LabRange(0, 255, 0, 140, 0, 120)],
    HoldColor.LIGHT_BLUE: [LabRange(100, 255, 0, 128, 0, 128)],
    HoldColor.NAVY: [LabRange(0, 80, 0, 140, 0, 110)],
    HoldColor.PURPLE: [LabRange(0, 255, 140, 255, 0, 120)],
    HoldColor.PINK: [LabRange(150, 255, 150, 255, 120, 160)],
    HoldColor.BROWN: [LabRange(0, 100, 130, 160, 130, 170)],
    HoldColor.WHITE: [LabRange(200, 255, 120, 136, 120, 136)],
    HoldColor.GRAY: [LabRange(80, 180, 120, 136, 120, 136)],
    HoldColor.BLACK: [LabRange(0, 50, 120, 136, 120, 136)],
}


def _copy_ranges(ranges: Mapping[HoldColor, List[LabRange]]) -> Dict[HoldColor, List[LabRange]]:
    return {color: [LabRange(**asdict(r)) for r in entries] for color, entries in ranges.items()}


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


def load_lab_ranges(config_path: str = DEFAULT_LAB_CONFIG_PATH) -> Dict[HoldColor, List[LabRange]]:
    path = Path(config_path)
    if not path.exists():
        return _copy_ranges(DEFAULT_LAB_RANGES)

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    loaded = _copy_ranges(DEFAULT_LAB_RANGES)
    for key, ranges in payload.items():
        try:
            color = parse_hold_color(key)
        except ValueError:
            continue
        loaded[color] = [LabRange(**entry).clamp() for entry in ranges]
    return loaded


def save_lab_ranges(
    ranges: Mapping[HoldColor, List[LabRange]],
    config_path: str = DEFAULT_LAB_CONFIG_PATH,
) -> None:
    serializable: MutableMapping[str, List[Dict[str, int]]] = {}
    for color, entries in ranges.items():
        serializable[color.value] = [asdict(entry.clamp()) for entry in entries]

    with Path(config_path).open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


class LabColorClassifier:
    def __init__(
        self,
        ranges: Mapping[HoldColor, List[LabRange]] | None = None,
        *,
        min_match_ratio: float = 0.03,
        chromatic_preference_ratio: float = 0.05,
    ):
        source = ranges if ranges is not None else DEFAULT_LAB_RANGES
        self.ranges = _copy_ranges(source)
        self.min_match_ratio = min_match_ratio
        self.chromatic_preference_ratio = chromatic_preference_ratio

    @classmethod
    def from_config(cls, config_path: str = DEFAULT_LAB_CONFIG_PATH) -> "LabColorClassifier":
        return cls(load_lab_ranges(config_path))

    def score_lab(self, lab_image: np.ndarray) -> Dict[HoldColor, int]:
        scores: Dict[HoldColor, int] = {}
        for color, entries in self.ranges.items():
            if not entries:
                scores[color] = 0
                continue
            mask = np.zeros(lab_image.shape[:2], dtype=np.uint8)
            for entry in entries:
                mask = cv2.bitwise_or(mask, entry.to_mask(lab_image))
            scores[color] = int(cv2.countNonZero(mask))
        return scores

    def classify_lab(self, lab_image: np.ndarray) -> HoldColor:
        total_pixels = int(lab_image.shape[0] * lab_image.shape[1])
        if total_pixels <= 0:
            return HoldColor.UNKNOWN

        scores = self.score_lab(lab_image)
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

    def classify_bgr(self, bgr_image: np.ndarray, mask: np.ndarray | None = None) -> HoldColor:
        if bgr_image.size == 0:
            return HoldColor.UNKNOWN
        retinex_image = apply_retinex(bgr_image)
        lab_image = cv2.cvtColor(retinex_image, cv2.COLOR_BGR2LAB)

        if mask is not None:
            valid = mask.astype(bool)
            selected = lab_image[valid]
            if selected.size == 0:
                return HoldColor.UNKNOWN
            # Evaluate only masked pixels to avoid background-biased color scores.
            lab_image = selected.reshape((-1, 1, 3))

        return self.classify_lab(lab_image)

    def score_bgr_ratios(self, bgr_image: np.ndarray, mask: np.ndarray | None = None) -> Dict[HoldColor, float]:
        if bgr_image.size == 0:
            return {color: 0.0 for color in self.ranges.keys()}

        retinex_image = apply_retinex(bgr_image)
        lab_image = cv2.cvtColor(retinex_image, cv2.COLOR_BGR2LAB)

        if mask is not None:
            valid = mask.astype(bool)
            selected = lab_image[valid]
            if selected.size == 0:
                return {color: 0.0 for color in self.ranges.keys()}
            lab_image = selected.reshape((-1, 1, 3))

        scores = self.score_lab(lab_image)
        total_pixels = int(lab_image.shape[0] * lab_image.shape[1])
        if total_pixels <= 0:
            return {color: 0.0 for color in scores.keys()}

        return {color: (count / total_pixels) for color, count in scores.items()}


def tune_lab_range(
    image_bgr: np.ndarray,
    classifier: LabColorClassifier,
    *,
    color: HoldColor,
    range_index: int = 0,
    config_path: str = DEFAULT_LAB_CONFIG_PATH,
) -> None:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Image is empty.")
    if range_index < 0:
        raise ValueError("range_index must be >= 0")

    ranges = classifier.ranges.setdefault(color, [])
    while len(ranges) <= range_index:
        ranges.append(LabRange(0, 255, 0, 255, 0, 255))

    base = ranges[range_index].clamp()
    
    retinex_image = apply_retinex(image_bgr)
    lab_image = cv2.cvtColor(retinex_image, cv2.COLOR_BGR2LAB)
    base_preview = image_bgr.copy()

    window_name = f"LAB Tuner - {color.value}[{range_index}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("L min", window_name, base.l_min, 255, lambda _: None)
    cv2.createTrackbar("L max", window_name, base.l_max, 255, lambda _: None)
    cv2.createTrackbar("A min", window_name, base.a_min, 255, lambda _: None)
    cv2.createTrackbar("A max", window_name, base.a_max, 255, lambda _: None)
    cv2.createTrackbar("B min", window_name, base.b_min, 255, lambda _: None)
    cv2.createTrackbar("B max", window_name, base.b_max, 255, lambda _: None)

    current = base
    print("LAB tuner controls: press 's' to save, 'q' or ESC to exit.")

    while True:
        current = LabRange(
            l_min=cv2.getTrackbarPos("L min", window_name),
            l_max=cv2.getTrackbarPos("L max", window_name),
            a_min=cv2.getTrackbarPos("A min", window_name),
            a_max=cv2.getTrackbarPos("A max", window_name),
            b_min=cv2.getTrackbarPos("B min", window_name),
            b_max=cv2.getTrackbarPos("B max", window_name),
        ).clamp()

        mask = current.to_mask(lab_image)
        masked = cv2.bitwise_and(base_preview, base_preview, mask=mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        preview = np.hstack([base_preview, mask_bgr, masked])

        matched = int(cv2.countNonZero(mask))
        total = int(mask.shape[0] * mask.shape[1])
        ratio = (matched / total) if total > 0 else 0.0

        cv2.putText(
            preview,
            f"{color.value}[{range_index}] - s:save q:quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            preview,
            f"match: {matched}/{total} ({ratio * 100:.2f}%)",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            ranges[range_index] = current
            save_lab_ranges(classifier.ranges, config_path=config_path)
            print(f"Saved {color.value}[{range_index}] to {config_path}: {asdict(current)}")

    ranges[range_index] = current
    cv2.destroyWindow(window_name)
