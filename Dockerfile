FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/var/task/yolo26l-seg.pt

COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY lambda_function.py ${LAMBDA_TASK_ROOT}/lambda_function.py
COPY yolo26l-seg.pt ${LAMBDA_TASK_ROOT}/yolo26l-seg.pt

CMD ["lambda_function.lambda_handler"]