import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from color_logic import HoldColor, color_to_bgr, parse_hold_color

DEFAULT_API_URL = "https://rwgwtrnbs4tsxmdlxu4r6bq2km0lkgyw.lambda-url.ap-northeast-2.on.aws/"
DEFAULT_CONF = 0.5
AUTHOR_NAME = "Jaehyun Choi"
GITHUB_URL = "https://github.com/linklingj/RouteFinder"
SAMPLE_IMAGES = {
    "example1": Path("input/example1.jpg"),
    "example2": Path("input/example2.jpg"),
}


def init_state() -> None:
    defaults = {
        "upload_token": "",
        "image_bytes": None,
        "image_bgr": None,
        "image_base64": "",
        "detections": None,
        "selected_hold_id": None,
        "selected_click": None,
        "last_click": None,
        "route_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_result_state() -> None:
    st.session_state.detections = None
    st.session_state.selected_hold_id = None
    st.session_state.selected_click = None
    st.session_state.last_click = None
    st.session_state.route_result = None


def set_input_image(file_bytes: bytes) -> None:
    upload_token = hashlib.md5(file_bytes).hexdigest()
    if upload_token == st.session_state.upload_token:
        return
    st.session_state.upload_token = upload_token
    st.session_state.image_bytes = file_bytes
    st.session_state.image_bgr = decode_image(file_bytes)
    st.session_state.image_base64 = image_to_base64(file_bytes)
    clear_result_state()


def decode_image(file_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("업로드한 파일을 이미지로 읽을 수 없습니다.")
    return image


def image_to_base64(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode("utf-8")


def parse_api_response(response: requests.Response) -> Dict[str, Any]:
    payload: Any = None
    raw_text = response.text.strip()
    content_type = response.headers.get("content-type", "")

    try:
        payload = response.json()
    except Exception:
        if raw_text:
            try:
                payload = json.loads(raw_text)
            except Exception:
                payload = None

    if payload is None:
        preview = raw_text[:300] if raw_text else "<empty>"
        raise RuntimeError(
            f"API 응답 JSON 파싱 실패 (status={response.status_code}, content-type={content_type}). "
            f"응답 본문: {preview}"
        )

    if isinstance(payload, str):
        text_payload = payload.strip()
        if text_payload:
            try:
                payload = json.loads(text_payload)
            except Exception:
                pass

    if isinstance(payload, dict) and "body" in payload:
        body = payload.get("body")
        status_code = int(payload.get("statusCode", response.status_code))
        if isinstance(body, str):
            try:
                body = json.loads(body) if body.strip() else {}
            except json.JSONDecodeError as exc:
                raise RuntimeError("API body JSON 파싱에 실패했습니다.") from exc
        elif body is None:
            body = {}
    elif isinstance(payload, dict):
        body = payload
        status_code = response.status_code
    else:
        raise RuntimeError("API 응답 형식이 예상과 다릅니다.")

    if status_code >= 400:
        message = body.get("error", f"API 요청 실패 (status={status_code})") if isinstance(body, dict) else f"API 요청 실패 (status={status_code})"
        raise RuntimeError(message)

    if isinstance(body, dict) and body.get("ok") is False:
        raise RuntimeError(body.get("error", "API 요청이 실패했습니다."))

    if not isinstance(body, dict):
        raise RuntimeError("API body 형식이 예상과 다릅니다.")
    return body


def call_api(api_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        api_url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120,
    )
    return parse_api_response(response)


def normalize_detections(detections: Any) -> List[Dict[str, Any]]:
    if not isinstance(detections, list):
        raise RuntimeError("응답 형식 오류: detections가 리스트가 아닙니다.")

    normalized = []
    for i, det in enumerate(detections):
        if not isinstance(det, dict):
            continue
        if "xyxy" not in det or not isinstance(det["xyxy"], list) or len(det["xyxy"]) != 4:
            continue
        copied = dict(det)
        copied["id"] = int(copied.get("id", i))
        copied["class_id"] = int(copied.get("class_id", 0))
        copied["confidence"] = float(copied.get("confidence", 0.0))
        copied["class_name"] = copied.get("class_name", "hold")
        copied["segment"] = copied.get("segment", [])
        normalized.append(copied)
    return normalized


def color_for_detection(detection: Dict[str, Any]) -> tuple[int, int, int]:
    color_name = detection.get("detected_color")
    if not color_name:
        color_name = detection.get("tape_color") if int(detection.get("class_id", -1)) == 9 else detection.get("hold_color")
    if not isinstance(color_name, str):
        return color_to_bgr(HoldColor.UNKNOWN)
    try:
        return color_to_bgr(parse_hold_color(color_name))
    except ValueError:
        return color_to_bgr(HoldColor.UNKNOWN)


def clamp_xyxy(xyxy: List[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def build_detection_mask(detection: Dict[str, Any], image_shape: tuple[int, int, int]) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    segment = detection.get("segment") or []
    if isinstance(segment, list) and len(segment) >= 3:
        pts = np.round(np.array(segment, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
        pts[:, :, 0] = np.clip(pts[:, :, 0], 0, width - 1)
        pts[:, :, 1] = np.clip(pts[:, :, 1], 0, height - 1)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    x1, y1, x2, y2 = clamp_xyxy(detection["xyxy"], width, height)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def draw_detections(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    selected_hold_id: int | None,
) -> np.ndarray:
    canvas = image_bgr.copy()
    overlay = canvas.copy()

    for det in detections:
        det_id = int(det.get("id", -1))
        color = color_for_detection(det)
        mask = build_detection_mask(det, canvas.shape)
        overlay[mask > 0] = color

        segment = det.get("segment") or []
        if isinstance(segment, list) and len(segment) >= 3:
            pts = np.round(np.array(segment, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
            thickness = 4 if det_id == selected_hold_id else 2
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)
            text_x, text_y = int(pts[0][0][0]), int(pts[0][0][1])
        else:
            x1, y1, x2, y2 = clamp_xyxy(det["xyxy"], canvas.shape[1], canvas.shape[0])
            thickness = 4 if det_id == selected_hold_id else 2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            text_x, text_y = x1, y1

        label = f"{det_id} {det.get('class_name', 'hold')}"
        cv2.putText(
            canvas,
            label,
            (text_x, max(18, text_y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def pick_hold_from_click(x: int, y: int, detections: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not detections:
        return None

    candidates: List[Dict[str, Any]] = []
    for det in detections:
        segment = det.get("segment") or []
        if isinstance(segment, list) and len(segment) >= 3:
            pts = np.array(segment, dtype=np.float32).reshape((-1, 1, 2))
            if cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0:
                candidates.append(det)

    if candidates:
        return max(candidates, key=lambda d: float(d.get("confidence", 0.0)))

    bbox_hits = []
    for det in detections:
        x1, y1, x2, y2 = det.get("xyxy", [0, 0, 0, 0])
        if x1 <= x <= x2 and y1 <= y <= y2:
            bbox_hits.append(det)
    if bbox_hits:
        return max(bbox_hits, key=lambda d: float(d.get("confidence", 0.0)))

    return min(
        detections,
        key=lambda d: (
            (((d["xyxy"][0] + d["xyxy"][2]) / 2.0) - x) ** 2
            + (((d["xyxy"][1] + d["xyxy"][3]) / 2.0) - y) ** 2
        ),
    )


def draw_route(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    route_result: Dict[str, Any] | None,
    selected_hold_id: int | None,
) -> np.ndarray:
    if not route_result:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    hold_ids = {int(item.get("detection_id", -1)) for item in route_result.get("holds", [])}
    tape_ids = {int(item.get("detection_id", -1)) for item in route_result.get("tapes", [])}

    for det in detections:
        det_id = int(det.get("id", -1))
        mask = build_detection_mask(det, canvas.shape)
        if det_id in hold_ids:
            canvas[mask > 0] = image_bgr[mask > 0]

            segment = det.get("segment") or []
            if isinstance(segment, list) and len(segment) >= 3:
                pts = np.round(np.array(segment, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], isClosed=True, color=(20, 220, 20), thickness=2)
            else:
                x1, y1, x2, y2 = clamp_xyxy(det["xyxy"], canvas.shape[1], canvas.shape[0])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 220, 20), 2)

            x1, y1, _, _ = clamp_xyxy(det["xyxy"], canvas.shape[1], canvas.shape[0])
            cv2.putText(canvas, det.get("class_name", "hold"), (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 220, 20), 2)

        if det_id in tape_ids:
            segment = det.get("segment") or []
            if isinstance(segment, list) and len(segment) >= 3:
                pts = np.round(np.array(segment, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], isClosed=True, color=(0, 200, 255), thickness=3)
            else:
                x1, y1, x2, y2 = clamp_xyxy(det["xyxy"], canvas.shape[1], canvas.shape[0])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 3)

    if selected_hold_id is not None:
        selected = next((d for d in detections if int(d.get("id", -1)) == selected_hold_id), None)
        if selected:
            x1, y1, x2, y2 = clamp_xyxy(selected["xyxy"], canvas.shape[1], canvas.shape[0])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 80, 80), 3)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def selected_hold_text(selected_hold: Dict[str, Any] | None) -> str:
    if not selected_hold:
        return "선택된 홀드가 없습니다."
    return (
        f"선택 홀드: id={selected_hold.get('id')} / "
        f"type={selected_hold.get('class_name')} / "
        f"color={selected_hold.get('detected_color', selected_hold.get('hold_color', 'unknown'))} / "
        f"conf={float(selected_hold.get('confidence', 0.0)):.2f}"
    )


def render_sample_icon(sample_key: str, sample_path: Path) -> None:
    if not sample_path.exists():
        st.caption(f"{sample_key}: 파일 없음")
        return
    image_b64 = base64.b64encode(sample_path.read_bytes()).decode("utf-8")
    st.markdown(
        (
            f"<a href='?sample={sample_key}'>"
            f"<img src='data:image/jpeg;base64,{image_b64}' "
            "style='width:100px;height:100px;object-fit:cover;border-radius:10px;border:1px solid #d1d5db;'/>"
            "</a>"
        ),
        unsafe_allow_html=True,
    )
    st.caption(sample_key)


def main() -> None:
    st.set_page_config(page_title="Route Finder", layout="wide")
    init_state()

    st.title("Route Finder")
    st.write("클라이밍장 이미지에서 홀드를 탐지하고, 특정 홀드를 선택하면 해당 홀드가 포함된 루트를 시각화합니다.")
    api_url = os.getenv("ROUTE_FINDER_API_URL", DEFAULT_API_URL).strip()
    sample_key = st.query_params.get("sample")
    if isinstance(sample_key, str) and sample_key in SAMPLE_IMAGES:
        sample_path = SAMPLE_IMAGES[sample_key]
        if sample_path.exists():
            set_input_image(sample_path.read_bytes())
        else:
            st.warning(f"샘플 이미지를 찾을 수 없습니다: {sample_path}")
        st.query_params.clear()

    left_col, right_col = st.columns([1, 1.35], gap="large")
    with left_col:
        st.subheader("입력")
        conf = st.number_input("Confidence", min_value=0.0, max_value=1.0, value=float(DEFAULT_CONF), step=0.01, format="%.2f")
        st.caption("예시 이미지 (아이콘 클릭 시 입력 이미지로 로드)")
        sample_col1, sample_col2 = st.columns(2)
        with sample_col1:
            render_sample_icon("example1", SAMPLE_IMAGES["example1"])
        with sample_col2:
            render_sample_icon("example2", SAMPLE_IMAGES["example2"])

        uploaded = st.file_uploader("이미지 업로드 (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            set_input_image(uploaded.getvalue())

        if st.session_state.image_bgr is not None:
            st.image(cv2.cvtColor(st.session_state.image_bgr, cv2.COLOR_BGR2RGB), caption="업로드 이미지", use_container_width=True)

        detect_clicked = st.button("홀드 탐지", use_container_width=True)
        route_clicked = st.button("선택한 홀드의 루트 찾기", use_container_width=True)

    if detect_clicked:
        if st.session_state.image_base64 == "":
            st.error("먼저 이미지를 업로드하세요.")
        elif not api_url:
            st.error("서버 URL 설정이 비어 있습니다. `ROUTE_FINDER_API_URL` 환경변수를 확인하세요.")
        else:
            try:
                result = call_api(
                    api_url,
                    {
                        "action": "predict",
                        "image_base64": st.session_state.image_base64,
                        "conf": conf,
                    },
                )
                detections = normalize_detections(result.get("detections"))
                st.session_state.detections = detections
                st.session_state.selected_hold_id = None
                st.session_state.selected_click = None
                st.session_state.last_click = None
                st.session_state.route_result = None
                st.success(f"홀드 탐지 완료: {len(detections)}개")
            except Exception as exc:
                st.error(str(exc))

    detections = st.session_state.detections
    selected_hold = None
    with right_col:
        st.subheader("결과")
        if isinstance(detections, list) and detections:
            st.caption("검출 결과 (이미지에서 홀드를 직접 클릭)")
            preview = draw_detections(st.session_state.image_bgr, detections, st.session_state.selected_hold_id)
            click = streamlit_image_coordinates(Image.fromarray(preview), key="hold-selector")

            if click and "x" in click and "y" in click:
                click_xy = (int(click["x"]), int(click["y"]))
                if click_xy != st.session_state.last_click:
                    st.session_state.last_click = click_xy
                    picked = pick_hold_from_click(click_xy[0], click_xy[1], detections)
                    if picked is not None:
                        st.session_state.selected_hold_id = int(picked.get("id", -1))
                        st.session_state.selected_click = {"x": click_xy[0], "y": click_xy[1]}
                        st.session_state.route_result = None
                    st.rerun()

            selected_hold = next((d for d in detections if int(d.get("id", -1)) == st.session_state.selected_hold_id), None)
            st.info(selected_hold_text(selected_hold))
        else:
            st.caption("탐지 결과가 아직 없습니다.")

    if route_clicked:
        if st.session_state.image_base64 == "":
            st.error("먼저 이미지를 업로드하세요.")
        elif not isinstance(st.session_state.detections, list):
            st.error("먼저 홀드 탐지를 실행하세요.")
        elif st.session_state.selected_click is None:
            st.error("먼저 이미지에서 홀드를 선택하세요.")
        elif not api_url:
            st.error("서버 URL 설정이 비어 있습니다. `ROUTE_FINDER_API_URL` 환경변수를 확인하세요.")
        else:
            try:
                result = call_api(
                    api_url,
                    {
                        "action": "find_route",
                        "image_base64": st.session_state.image_base64,
                        "click": st.session_state.selected_click,
                        "conf": conf,
                    },
                )
                detections = normalize_detections(result.get("detections"))
                route_result = result.get("route")
                if route_result is None:
                    raise RuntimeError("선택 홀드를 기준으로 루트를 찾지 못했습니다.")

                st.session_state.detections = detections
                st.session_state.route_result = route_result
                st.success("루트 탐색 완료")
            except Exception as exc:
                st.error(str(exc))

    with right_col:
        if st.session_state.route_result:
            st.subheader("루트 시각화")
            route_preview = draw_route(
                st.session_state.image_bgr,
                st.session_state.detections,
                st.session_state.route_result,
                st.session_state.selected_hold_id,
            )
            st.image(route_preview, caption="선택 루트 강조 (기타 홀드 흑백)", use_container_width=True)

            route = st.session_state.route_result
            st.write(f"선택 홀드 색상: `{route.get('hold_color', 'unknown')}`")
            if route.get("tape_color"):
                st.write(f"테이프 색상: `{route.get('tape_color')}`")

            st.write("루트 홀드 종류 목록")
            hold_type_counts = route.get("hold_type_counts", {})
            if isinstance(hold_type_counts, dict) and hold_type_counts:
                for hold_name, count in hold_type_counts.items():
                    st.write(f"- {hold_name}: {count}")
            else:
                st.write("- 없음")

            st.write("루트 홀드 리스트 (홀드 종류)")
            route_holds = route.get("holds", [])
            if isinstance(route_holds, list) and route_holds:
                for i, hold in enumerate(route_holds, start=1):
                    st.write(f"{i}. {hold.get('class_name', 'unknown')}")
            else:
                st.write("1. 없음")

    st.markdown("---")
    st.markdown(f"제작자: **{AUTHOR_NAME}**")
    st.markdown(f"GitHub: [{GITHUB_URL}]({GITHUB_URL})")


if __name__ == "__main__":
    main()
