FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Install OS dependencies required by scientific/geospatial Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    pkg-config \
    cmake \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libnetcdf-dev \
    libeccodes-dev \
    libeccodes-tools \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libgdal-dev \
    gdal-bin \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 11.8 wheels) first, then install project dependencies.
COPY pyproject.toml README.md /workspace/
COPY physicsnemo /workspace/physicsnemo
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

COPY . /workspace
RUN pip install -e .

CMD ["bash"]