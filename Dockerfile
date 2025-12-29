# 使用官方推荐的基础镜像
FROM python:3.11-slim

WORKDIR /

# 1. 安装系统级依赖
# 必须保留 git-lfs，因为 handler.py 需要它在首次运行时将完整模型拉取到网络卷
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

# 3. 安装您指定的核心 PyTorch 版本 (保持 CUDA 12.6 源)
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# 4. 安装业务和模型依赖 (完全保留您的版本要求)
RUN pip install --no-cache-dir \
    numpy librosa scipy phonemizer textgrid einops transformers \
    huggingface_hub munch pyyaml tensorboard colorlog omegaconf \
    unidecode inflect jieba pypinyin pydantic gradio fastapi uvicorn \
    python-dotenv torchtune torchao==0.13.0 \
    'websockets>=13.0' supabase boto3 requests orjson \
    runpod 

# 5. 复制仓库代码并安装
# 此时不进行任何模型克隆，确保构建过程飞快，避免 30 分钟超时限制
COPY . /app
WORKDIR /app
RUN pip install --no-deps -e .

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 关键路径配置：MODEL_DIR 必须指向 RunPod 网络卷的固定挂载路径
# 在 Serverless 环境中，附加的网络卷始终挂载在 /runpod-volume
ENV MODEL_DIR=/runpod-volume/FireRedTTS2

# 启动 Worker
CMD ["python", "-u", "handler.py"]
