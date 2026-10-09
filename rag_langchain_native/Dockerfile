# V3 独立镜像；从仓库根目录构建：docker build -f rag_langchain_native/Dockerfile .
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    V3_ENVIRONMENT=production

RUN groupadd --system --gid 10001 rag \
    && useradd --system --uid 10001 --gid rag --home /app rag
WORKDIR /app

COPY rag_langchain_native/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --requirement /tmp/requirements.txt

COPY --chown=rag:rag rag_langchain_native /app/rag_langchain_native
RUN mkdir -p /app/rag_langchain_native/runtime \
    && chown -R rag:rag /app/rag_langchain_native/runtime

USER rag
EXPOSE 8011
VOLUME ["/app/rag_langchain_native/runtime"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8011/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "rag_langchain_native.api:app", "--host", "0.0.0.0", "--port", "8011", "--workers", "1", "--timeout-graceful-shutdown", "30"]
