FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV SERVER_URL=http://localhost:8080
ENV HOSPITAL_NAME=hospital_1
ENV DATA_PATH=/data
ENV LOCAL_EPOCHS=2
ENV BATCH_SIZE=16
ENV LEARNING_RATE=0.001

CMD ["python", "src/main.py"]
