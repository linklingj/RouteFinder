import cv2
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


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
    PURPLE = "purple"
    PINK = "pink"
    WHITE = "white"
    BLACK = "black"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HueRange:
    color: HoldColor
    # hue 범위(도 단위, 0~359)
    min_deg: int
    max_deg: int

    def contains(self, hue_deg: int) -> bool:
        if self.min_deg <= self.max_deg:
            return self.min_deg <= hue_deg <= self.max_deg
        return hue_deg >= self.min_deg or hue_deg <= self.max_deg


# 필요하면 이 테이블 숫자만 바꿔서 쉽게 범위를 조정하면 된다.
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
    hue_scale: str = "opencv",
    hue_ranges: Tuple[HueRange, ...] = DEFAULT_HUE_RANGES,
) -> HoldColor:
    """
    hue를 HoldColor enum으로 변환.
    hue_scale:
      - "opencv": 0~179 (OpenCV HSV)
      - "degree": 0~359
    """
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
    hold_color: HoldColor = HoldColor.UNKNOWN

@dataclass
class Tape:
    confidence: float
    xyxy: List[float]
    tape_color: HoldColor = HoldColor.UNKNOWN

class ImageInfo:
    img : cv2.Mat
    holds: List[Hold]
    tapes: List[Tape]

    def __init__(self, img: cv2.Mat, detections: List[Dict]):
        self.img = img
        for det in detections:
            if det["class_id"] == 10:
                xyxy = [float(v) for v in det["xyxy"]]
                self.tapes.append(
                    Tape(
                        confidence=float(det["confidence"]),
                        xyxy=xyxy,
                        tape_color=self.get_color(xyxy),
                    )
                )
            else:
                xyxy = [float(v) for v in det["xyxy"]]
                self.holds.append(
                    Hold(
                        hold_type=int(det["class_id"]),
                        confidence=float(det["confidence"]),
                        xyxy=[float(v) for v in det["xyxy"]],
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
        mean_hue = int(hue_channel.mean())
        return hue_to_hold_color(mean_hue)
    
        

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

    def set_route(self, img_info: ImageInfo, ref_hold_pos):
        # ref_hold_pos에서 가장 가까운 홀드를 찾고
        # 그 홀드의 색상을 route의 hold_color로 설정
        # 같은 색의 홀드들을 route의 holds에 추가

        if not img_info.holds:
            return False
        min_hold_dist, closest_hold = float("inf"), None
        for hold in img_info.holds:
            hold_center = (
                (hold.xyxy[0] + hold.xyxy[2]) / 2,
                (hold.xyxy[1] + hold.xyxy[3]) / 2,
            )
            dist = ((hold_center[0] - ref_hold_pos[0]) ** 2 + (hold_center[1] - ref_hold_pos[1]) ** 2) ** 0.5
            if dist < min_hold_dist:
                min_hold_dist, closest_hold = dist, hold

        if closest_hold:
            self.hold_color = closest_hold.hold_color.value
            for hold in img_info.holds:
                if hold.hold_color == closest_hold.hold_color:
                    self.add_hold(hold)
        else:
            return False

        # 시작홀드에서 가장 가까운 테이프를 찾아서 route의 start_tape로 설정
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

        print(f"Route set: hold_color={self.hold_color}, tape_color={self.tape_color}, holds={len(self.holds)}, start_tape={self.start_tape is not None}")
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
