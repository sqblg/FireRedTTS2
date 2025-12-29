# 保持你指定的 python:3.11-slim，不动版本
FROM python:3.11-slim

WORKDIR /

# 1. 安装系统级依赖 (保持你的列表)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    espeak-ng \
    git-lfs \
    build-essential \
    cmake \
    g++ \
    wget && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# 2. 升级基础工具
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. 安装核心 PyTorch (严格保留你的版本 2.7.1)
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# 4. 安装业务依赖 (严格保留你的版本，包括 torchao==0.13.0)
RUN pip install --no-cache-dir \
    numpy librosa scipy phonemizer textgrid einops transformers \
    huggingface_hub munch pyyaml tensorboard colorlog omegaconf \
    unidecode inflect jieba pypinyin pydantic gradio fastapi uvicorn \
    python-dotenv torchtune torchao==0.13.0 \
    'websockets>=13.0' supabase boto3 requests orjson \
    runpod 

# 5. 复制仓库代码并安装
# COPY . /app 会把当前目录(包括 handler.py 和 assets) 复制进去
COPY . /app
WORKDIR /app
RUN pip install --no-deps -e .

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 🔴 关键配置：指定模型在网络卷中的位置
ENV MODEL_DIR=/runpod-volume/FireRedTTS2
# 🔴 关键配置：指定资产在容器中的位置
ENV ASSETS_DIR=/app/assets
ENV PROMPT_TEXTS_FILE=/app/prompt_texts.json

# 6. 启动命令 (修正为你的文件名 handler.py)
CMD ["python", "-u", "/app/handler.py"]
