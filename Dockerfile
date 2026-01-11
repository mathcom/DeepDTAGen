# 1. Triton 호환 베이스 이미지 (Ubuntu 22.04 + CUDA 11.7)
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 기본 환경 설정 (타임존 등 인터랙션 방지)
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# 2. 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    libxrender1 \
    libxext6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 3. Miniforge 설치 (Miniconda의 약관 문제를 피하기 위한 오픈소스 대안)
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh && \
    bash miniforge.sh -b -p /opt/conda && \
    rm miniforge.sh

# 4. 가상환경 생성 (conda-forge 채널만 사용하여 약관 에러 방지)
RUN conda create -n dta_env python=3.8 -y
ENV CONDA_DEFAULT_ENV=dta_env
ENV PATH=${CONDA_DIR}/envs/dta_env/bin:${PATH}

# 5. Pip 업그레이드 및 PyTorch 설치 (CUDA 11.7 전용)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# 6. 데이터 과학 도구 및 RDKit 설치
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    pandas \
    scikit-learn \
    scipy \
    tqdm \
    rdkit

# 7. PyTorch Geometric (PyG) 설치
# PyG 의존성 설치
RUN pip install --no-cache-dir \
    torch-scatter==2.1.0 \
    torch-sparse==0.6.16 \
    torch-cluster==1.6.0 \
    torch-spline-conv==1.2.1 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

# PyG 설치
RUN pip install --no-cache-dir torch-geometric==2.2.0

# 8. 나머지 라이브러리 (Fairseq, Transformers 등)
RUN pip install --no-cache-dir \
    fairseq==0.10.2 \
    selfies \
    einops==0.6.0 \
    mordred \
    transformers \
    sentence-transformers \
    sentencepiece \
    networkx \
    matplotlib \
    seaborn \
    protobuf==3.20.3 \
    tensorboard \
    jupyterlab \
    ipywidgets

# 9. Triton 배포용 도구
RUN pip install conda-pack

EXPOSE 8888

CMD ["bash"]