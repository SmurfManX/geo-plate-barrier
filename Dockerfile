FROM python:3.14-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore \
    HOME=/home/app

# libgl/glib for OpenCV, libv4l for USB cameras on Linux, ffmpeg libs come with opencv wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libv4l-0 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 app \
 && mkdir /data && chown app:app /data

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

USER app
# Bake the ONNX models into the image so the container starts offline.
ARG ALPR_DETECTOR=yolo-v9-t-512-license-plate-end2end
ARG ALPR_OCR=cct-s-v2-global-model
ENV ALPR_DETECTOR=${ALPR_DETECTOR} ALPR_OCR=${ALPR_OCR}
RUN python -c "import os; from fast_alpr import ALPR; ALPR(detector_model=os.environ['ALPR_DETECTOR'], ocr_model=os.environ['ALPR_OCR'], ocr_device='cpu')"

COPY --chown=app:app app ./app
COPY --chown=app:app tests ./tests

ENV DB_PATH=/data/barrier.db
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--no-access-log"]

# Test/benchmark image: docker build --target bench -t plate-barrier:bench .
FROM app AS bench
USER root
RUN apt-get update && apt-get install -y --no-install-recommends fonts-liberation \
 && rm -rf /var/lib/apt/lists/* && pip install pillow==13.0.0 pytest
USER app
