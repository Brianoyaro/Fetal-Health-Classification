FROM python:3.13.5-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./
RUN uv sync --locked

COPY app.py ./
COPY models/xgboost_pipeline.joblib models/xgboost_pipeline.joblib


EXPOSE 9696

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]


# docker build -t fetal-health-classification .
# docker run -it --rm -p 9696:9696 fetal-health-classification