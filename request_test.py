import base64
import json
import requests
from pathlib import Path

# ==============================
# 설정
# ==============================
IMAGE_PATH = "input/theclimb1.jpeg"  # 테스트 이미지 경로
LAMBDA_URL = "https://7bikf37uxfvble7avhpcvfn46q0omusw.lambda-url.ap-northeast-2.on.aws/"  # Function URL 넣기
CONF = 0.5


# ==============================
# 1. 이미지 → base64
# ==============================
def image_to_base64(image_path: str) -> str:
    image_bytes = Path(image_path).read_bytes()
    return base64.b64encode(image_bytes).decode("utf-8")


# ==============================
# 2. 요청 보내기
# ==============================
def send_request(image_base64: str):
    payload = {
        # "action": "health",
        "image_base64": image_base64,
        "conf": CONF
    }
    

    response = requests.post(
        LAMBDA_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    return response


# ==============================
# 3. 실행
# ==============================
def main():
    print("🔄 Encoding image...")
    img_b64 = image_to_base64(IMAGE_PATH)

    print("🚀 Sending request...")
    response = send_request(img_b64)

    print("\n📦 Raw Response:")
    print(response.text)
    print(response.status_code)

    # Lambda는 body 안에 JSON 문자열이 들어있을 수 있음
    try:
        data = response.json()

        if "body" in data:
            body = json.loads(data["body"])
        else:
            body = data

        print("\n✅ Parsed Result:")
        print(json.dumps(body, indent=2))

    except Exception as e:
        print("❌ Failed to parse response:", e)


if __name__ == "__main__":
    main()