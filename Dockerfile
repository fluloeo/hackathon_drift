FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .
COPY models/ ./models/

RUN mkdir -p /data/input /data/output

CMD ["python", "predict.py"]