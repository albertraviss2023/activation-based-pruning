FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY ui/requirements.txt ./ui/requirements.txt

RUN pip install --upgrade pip \
    && pip install -r ui/requirements.txt \
    && pip install -e .

COPY ui ./ui
COPY custom_methods ./custom_methods

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
