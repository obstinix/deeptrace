FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

RUN mkdir -p /home/user/data && chown -R user:user /home/user/data

WORKDIR /app

COPY requirements-deploy.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user src/ src/
COPY --chown=user api/ api/
COPY --chown=user training/configs/ training/configs/
COPY --chown=user celery_app.py celery_app.py
COPY --chown=user worker/ worker/
COPY --chown=user checkpoints/ checkpoints/
COPY --chown=user start-deploy.sh start-deploy.sh

RUN chmod +x start-deploy.sh && chown user:user start-deploy.sh
ENV PYTHONPATH=/app/src:/app

USER user

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["./start-deploy.sh"]


