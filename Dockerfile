# SE374.R11 BCSS-WSSS Clinical Digital Pathology Workstation Container
# Serves FastAPI + offline HTML5 Canvas viewer on port 8000 with /health probe.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/tmp/hf_cache \
    WSSS_GATE7_DIR=/app/checkpoints/vps_evidence

WORKDIR /app

# Minimal native runtime libraries required by OpenCV headless and Pillow
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (pywin32 is skipped automatically on Linux via marker)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy modular application source and static clinical viewer assets
COPY src /app/src

# Prepare volume mountpoints for checkpoints and real histopathology cohorts
RUN mkdir -p /app/checkpoints /app/data \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5); assert r.status == 200" || exit 1

CMD ["uvicorn", "src.visualization.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
