FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/fontconfig && chmod 777 /tmp/fontconfig
RUN mkdir -p /tmp/huggingface_cache && chmod 777 /tmp/huggingface_cache
RUN mkdir -p /tmp/xdg_cache && chmod 777 /tmp/xdg_cache

ENV FONTCONFIG_PATH=/tmp/fontconfig
ENV XDG_CACHE_HOME=/tmp/xdg_cache
ENV HF_HOME=/tmp/huggingface_cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]