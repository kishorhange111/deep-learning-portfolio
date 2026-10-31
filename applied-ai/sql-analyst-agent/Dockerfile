# ShopCo SQL analyst agent: web app image.
#   docker build -t shopco-agent .
#   docker run -p 8080:8080 -e ANTHROPIC_API_KEY=... shopco-agent
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py tasks.json ./
COPY static ./static
# build the synthetic database at image build time (deterministic, ~1 s)
RUN python make_db.py
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
ENV PORT=8080
EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/api/health')"
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
