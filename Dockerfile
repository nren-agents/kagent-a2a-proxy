FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Copy application code (full package directory)
COPY kagent_a2a_proxy ./kagent_a2a_proxy/


# Sync again to install the project itself
RUN uv sync --no-dev

ENV PYTHONUNBUFFERED=1 \
    PROXY_LOG_LEVEL=info

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz/ready')"

CMD ["uv", "run", "uvicorn", "kagent_a2a_proxy.main:app", \
     "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
