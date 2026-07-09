FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*
COPY requirements-deploy.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY api/ api/
COPY training/configs/ training/configs/
COPY celery_app.py celery_app.py
COPY worker/ worker/
COPY checkpoints/ checkpoints/
ENV PYTHONPATH=/app/src
COPY start-deploy.sh start-deploy.sh
RUN chmod +x start-deploy.sh
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/api/health || exit 1
CMD ["./start-deploy.sh"]

