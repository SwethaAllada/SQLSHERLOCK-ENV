FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# Install Python dependencies first so this layer is cached
COPY sqlsherlock_env/server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire repo
COPY . .

EXPOSE 7860

# PYTHONPATH so "from models import ..." and "from server.xxx import ..." resolve correctly
ENV PYTHONPATH=/app/sqlsherlock_env

# Health check — must pass before HF Spaces routes traffic
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
  --retries=3 CMD curl -f http://localhost:7860/health || exit 1

# Run from sqlsherlock_env/ so relative module paths match the import structure
# Use 1 worker to stay within 2 vCPU / 8 GB RAM constraints
WORKDIR /app/sqlsherlock_env
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", \
     "--port", "7860", "--workers", "1"]
