FROM public.ecr.aws/lambda/python:3.11

# 작업 디렉토리
WORKDIR ${LAMBDA_TASK_ROOT}

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY lambda_function.py .
COPY infer.py .
COPY model_loader.py .
COPY color_logic.py .
COPY lab_ranges.json .
COPY models ./models

# Lambda 핸들러 지정
CMD ["lambda_function.lambda_handler"]
