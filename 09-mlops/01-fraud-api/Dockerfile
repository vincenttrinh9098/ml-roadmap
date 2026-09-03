# Dockerfile
#
# Stage 9: package app.py + your trained model into a self-contained image that runs
# identically on any machine with Docker installed -- your laptop, a teammate's laptop,
# or a cloud server.
#
# Build with:   docker build -t fraud-api .
# Run with:     docker run -p 8000:8000 fraud-api
#
# Each instruction below (FROM, COPY, RUN, ...) creates one cached "layer" -- Docker only
# re-runs a layer (and everything after it) if that layer's inputs actually changed. This is
# why requirements.txt is copied and installed BEFORE the rest of your code: your code
# changes constantly, your dependencies rarely do -- so ordering it this way means most
# rebuilds skip the slow "install everything" step entirely and reuse the cached layer.

FROM python:3.12-slim

WORKDIR /app

# Copy dependencies first so Docker can cache this layer
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and trained model files
COPY app.py .
COPY fraud_model.joblib .
COPY scaler.joblib .

EXPOSE 8000

# Start FastAPI
ENV PORT=8000
CMD uvicorn app:app --host 0.0.0.0 --port $PORT