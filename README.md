# RouteFinder
> 딥러닝 기반 클라이밍 홀드/테이프 인식 및 루트 찾기

![Badge](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square)
![Badge](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=flat-square)
![Badge](https://img.shields.io/badge/Ultralytics-YOLOv8-111111?style=flat-square)
![Badge](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?style=flat-square)
![Badge](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=flat-square)
![Badge](https://img.shields.io/badge/AWS-Lambda-232F3E?style=flat-square)

## 개요
RouteFinder는 **Computer Vision + Deep Learning**으로 클라이밍 벽 이미지에서 홀드(및 테이프)를 세그멘테이션하고, 사용자가 클릭한 홀드를 기준으로 등반 루트를 자동 추출하는 프로젝트입니다.

> -> [데모 페이지 링크](https://climbingroutefinder-9whx6sena2qj7kyjccs6ea.streamlit.app/) <-

개발 포인트
- 세그멘테이션 기반 객체 인식 파이프라인 구축
- 조명 변화에 강인한 색상 분류(LAB + Retinex) 적용
- 추론 API(AWS Lambda)와 UI(Streamlit) 분리 배포
- 추론 결과를 사람 중심 인터랙션(클릭 기반 루트 탐색)으로 연결

![image](demo/result.gif)


## 데모

사용 흐름
1. 이미지 업로드
2. 홀드 탐지 실행
3. 이미지에서 특정 홀드 클릭
4. 선택한 홀드와 같은 색상의 루트 자동 시각화

![image](demo/web.png)

## 기능
- **홀드 세그멘테이션 추론**: YOLO 기반 모델로 홀드/테이프 검출
- **색상 분류**: Retinex 전처리 + LAB 범위 기반 색상 추정
- **클릭 기반 루트 탐색**: 선택 홀드와 동일 색상 홀드 군집 추출
- **테이프 보조 판단**: 인접 테이프 색과 위치 분포로 시작/종료 후보 추론
- **실시간 시각화**: 선택 루트만 컬러 강조, 나머지는 흑백 처리
- **서버-클라이언트 분리**: Lambda API와 Streamlit 프론트 분리 구성

이미지 자리: 기능 요약 다이어그램

## 아키텍처
구성 요소
- `train.py`: Ultralytics YOLO 학습 엔트리
- `model_loader.py`: 단일 모델 인스턴스 로딩(재사용)
- `infer.py`: 기본 추론/세그멘트 직렬화
- `color_logic.py`: Retinex + LAB 기반 색 분류
- `lambda_function.py`: API 핸들러(`predict`, `find_route`, `health`)
- `streamlit_app.py`: 사용자 인터페이스 및 인터랙션

데이터 플로우
1. Streamlit이 이미지를 base64로 인코딩 후 API 호출
2. Lambda가 이미지 디코딩 후 YOLO 추론 수행
3. 세그멘트/박스/클래스 + 색상 라벨 생성
4. `find_route` 요청 시 클릭 좌표 기준 루트 계산
5. 프론트에서 루트 강조 렌더링

아키텍처 의사결정
- **YOLO 세그멘테이션 선택**
  - 장점: 객체 탐지 + 마스크를 단일 파이프라인으로 처리 가능
  - 트레이드오프: 초경량 모델 대비 추론 비용 증가 가능
- **LAB 규칙 기반 색 분류 병행**
  - 장점: 데이터가 제한된 환경에서도 안정적으로 색상 기준 제어 가능
  - 트레이드오프: 조명/카메라 도메인 변화가 크면 수동 튜닝 필요
- **Lambda 서버리스 배포**
  - 장점: 운영 부담이 낮고 API 엔드포인트 관리가 단순함
  - 트레이드오프: 콜드스타트/메모리 제약으로 지연 가능성 존재
- **프론트/백엔드 분리**
  - 장점: UI 실험과 추론 API를 독립적으로 개발/배포 가능
  - 트레이드오프: 네트워크 왕복 지연과 응답 파싱 복잡도 증가

```mermaid
flowchart LR
    U[사용자]
    B[브라우저]
    F[Streamlit Frontend<br/>streamlit_app.py]
    A[AWS Lambda API<br/>lambda_function.py]
    M[YOLO Segmentation<br/>model_loader.py + infer.py]
    C[Color Classification<br/>color_logic.py]
    R[Route Builder<br/>click 기반 루트 탐색]
    P[결과 시각화<br/>오버레이 + 루트 강조]

    U --> B --> F
    F -->|action: predict/find_route<br/>image_base64 + conf + click| A
    A --> M
    M -->|detections + masks| C
    C --> R
    R -->|detections + route JSON| A
    A --> F
    F --> P --> B
```

## 데이터
- 데이터는 [roboflow](https://app.roboflow.com/panclimb/climbing-segmentation-lruqt)에서 직접 라벨링
-  한국 클라이밍장 도메인에 맞게 직접 촬영한 이미지 사용
### 전처리
- 색상 분류 전 Retinex(MSRCR 계열) 적용으로 조명 편차 완화
![image](demo/retinex.png)
- LAB 색공간으로 변환 후 `lab_ranges.json` 기준 매칭
  - HSV 대신 LAB를 채택해 조명 변화 민감도를 낮춤
- 색별 LAB 범위 조정할 수 있는 유틸 프로그램 `color_logic.py` 제작
![image](demo/color_tune.png)


## 모델
- 프레임워크: Ultralytics YOLO (Segmentation)
- 학습 스크립트: `train.py`

학습 파라미터(현재 코드 기준)
- `epochs=100`
- `imgsz=780`
- `dropout=0.2`

![image](demo/holds.png)

## 학습 평가

### Detection/Segmentation:

  Evaluation Metrics (Valid Set)

| Metric     | Value  |
|-----------|--------|
| mAP@50    | 55.4%  |
| Precision | 67.6%  |
| Recall    | 48.9%  |
| F1 Score  | 56.8%  |
### 보조 지표
홀드 색상 분류 정확도
  
| Metric     | Value  |
|-----------|--------|
| holds    | 71.3%  |
| tapes    | 68.9%  |
| combine    | 70.5%  |
### 사용자 관점 지표
클릭 기준 루트 추론 성공률
  
| Metric     | Value  |
|-----------|--------|
| success rate    | 80%  |

### 의사결정 및 트레이드오프
- 단순 mAP만으로는 실제 사용자 경험(루트 탐색 성공 여부)을 충분히 설명하기 어렵기 때문에,
  서비스 지표를 함께 관리하는 전략을 사용

![image](demo/confusion.png)

- 클라이밍 홀드 분류 문제 특성상 false-negative를 최소화하는 것이 중요하다고 판단해 낮은 conf를 선택하는 것이 유리

![image](demo/f1-curve.png)

## 오류 분석

본 프로젝트는 실제 클라이밍 환경(조명 변화, 작은 홀드, 겹침 등)을 반영하기 때문에 단순 mAP 외에도 다양한 오류 패턴이 존재합니다. 주요 오류 유형과 개선 방향은 다음과 같습니다.

#### 1) 작은 홀드 탐지 문제
- 문제: 원거리 촬영 이미지에서 foothold와 같은 작은 홀드가 누락되는 경우가 빈번
- 원인: 학습 데이터에서 작은 객체 비율 부족, YOLO의 기본 stride 구조에서 작은 객체 표현 한계
- 개선:
  - 입력 해상도(`imgsz`) 증가
  - 작은 홀드 샘플을 별도로 수집하여 데이터 불균형 완화
  - false-negative 사례를 수집하여 hard example 재학습

#### 2) 겹쳐있는 홀드 분류 오류
- 문제: 겹쳐있는 홀드가 하나의 객체로 합쳐지거나 잘못된 클래스로 분류됨
- 원인: annotation 단계에서 경계 기준 불일치, 겹침 상황에 대한 데이터 부족
- 개선:
  - 라벨링 가이드라인 재정의 (겹침 시 분리 기준 명확화)
  - NMS 및 후처리 단계에서 mask 기반 IoU filtering 적용

#### 3) 색상 분류 오류
- 문제: 동일 색상 홀드가 조명에 따라 다른 색으로 분류
- 원인: 조명 변화 및 카메라 화이트밸런스 영향, HSV 기반 단순 분류의 한계
- 개선 방안:
  - Retinex(MSRCR) 기반 전처리로 조명 영향 최소화
  - LAB 색공간 기반 거리 계산으로 색상 안정성 확보
  - 색상 범위(`lab_ranges.json`) 지속적 튜닝
- todo: KD-tree 기반 nearest color mapping 방식 적용 고려

#### 4) 일반화 성능 문제
- 문제: 특정 클라이밍장에서는 잘 작동하지만 새로운 환경에서 성능 저하
- 원인: 데이터 도메인 편향 (조명, 벽 색상, 촬영 각도)
- 개선:
  - 다양한 체육관 데이터 추가
  - 색상/조명 augmentation 강화

## 배포
### 1) API 배포 (AWS Lambda)
- 컨테이너: `Dockerfile` (Python 3.11 Lambda 베이스)
- 핸들러: `lambda_function.lambda_handler`
- 주요 액션
  - `predict`: 홀드/테이프 검출 + 색상 주석
  - `find_route`: 클릭 좌표 기반 루트 계산
  - `health`: 상태 확인

### 2) 프론트 배포 (Streamlit)
- 엔트리: `streamlit_app.py`
- 환경변수: `ROUTE_FINDER_API_URL`
- 로컬 실행 예시
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

의사결정 및 트레이드오프
- Streamlit을 사용해 프로토타입 속도를 높였고,
  커스텀 UX 요구가 커질 경우 React/Next.js로 이관하는 확장 전략을 고려할 수 있습니다.

```mermaid
flowchart TB
    subgraph Dev[개발 환경]
        D1[학습 서버/로컬 GPU]
        D2[train.py]
        D3[모델 아티팩트<br/>models/0314-2.pt]
        D1 --> D2 --> D3
    end

    subgraph AWS[배포: AWS]
        ECR[ECR 이미지]
        L[AWS Lambda 컨테이너<br/>Python 3.11]
        URL[Lambda Function URL]
        ECR --> L --> URL
    end

    subgraph FE[프론트 배포]
        S[Streamlit 앱<br/>streamlit_app.py]
    end

    Repo[GitHub Repository] --> ECR
    Repo --> S
    D3 --> ECR
    User[사용자 브라우저] --> S
    S -->|HTTPS API 호출| URL
    URL -->|JSON 응답| S
```

### 활용 방안

본 프로젝트는 단순 홀드 인식 기능을 넘어 다양한 클라이밍 관련 서비스로 확장될 수 있습니다.

#### 1) 루트 종류 자동 분류
- 색상, 위치, 연결 패턴을 기반으로 루트를 자동으로 분류
  - 다이내믹 / 밸런스 / 오버행 루트 구분

#### 2) 루트 기록 및 저장 애플리케이션
- 사용자가 클라이밍 후 자신이 완등한 루트를 기록

#### 3) 루트 추천 시스템
- 특정 사용자 수준에 맞는 루트를 자동 추천

#### 4) 루트 세팅 지원 AI
- 새로운 루트를 만들 때 AI가 도움 제공
- 클라이밍장 운영자/루트세터에게 유용

#### 5) 실시간 코칭 시스템
- 사람 pose estimation과 결합 시: 동작 피드백 제공

---

### 빠른 시작
```bash
# 1) 의존성 설치
pip install -r requirements_project.txt

# 2) 로컬 추론 테스트 (예시)
python request_test.py

# 3) 프론트 실행
streamlit run streamlit_app.py
```
