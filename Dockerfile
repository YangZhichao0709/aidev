# 安定版の 2.0.1 に戻す
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

WORKDIR /app
ENV REBUILD_DATE=20251205-fix

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "-u", "handler.py" ]