# 使用 Python 3.11 作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /

# 1. 安装系统级依赖
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    espeak-ng \
    git-lfs \
    build-essential \
    cmake \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. 升级基础工具
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. 安装您指定的核心 PyTorch 版本 (CUDA 12.6 源)
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# 4. 安装业务和模型依赖 (完全保留您要求的版本)
RUN pip install --no-cache-dir \
    numpy \
    librosa \
    scipy \
    phonemizer \
    textgrid \
    einops \
    transformers \
    huggingface_hub \
    munch \
    pyyaml \
    tensorboard \
    colorlog \
    omegaconf \
    unidecode \
    inflect \
    jieba \
    pypinyin \
    pydantic \
    gradio \
    fastapi \
    uvicorn \
    python-dotenv \
    torchtune \
    torchao==0.13.0 \
    'websockets>=13.0' \
    supabase \
    boto3 \
    requests \
    orjson \
    runpod  # 必须安装 RunPod SDK 用于 Serverless 环境

# 5. 设置模型预下载 (利用构建阶段缓存模型，减少冷启动时间)
RUN git lfs install && \
    mkdir -p /models/FireRedTTS2 && \
    cd /models && \
    git clone https://huggingface.co/FireRedTeam/FireRedTTS2 && \
    cd FireRedTTS2 && git lfs pull

# 6. 将当前仓库（GitHub 代码）复制到镜像中并安装主程序
COPY . /app
WORKDIR /app
RUN pip install --no-deps -e .

# 设置环境变量，确保 Python 输出能即时显示在日志中
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/models/FireRedTTS2

# 启动 Worker
CMD ["python", "-u", "handler.py"]
