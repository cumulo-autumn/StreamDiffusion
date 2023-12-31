# build frontend with node
FROM node:20-alpine AS frontend
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY frontend .
RUN npm i -g pnpm
RUN pnpm i
RUN pnpm run build

# build backend on CUDA
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 AS backend
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_MAJOR=20

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    # python build dependencies \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER root

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

RUN curl https://pyenv.run | bash
ENV PATH=$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH
ARG PYTHON_VERSION=3.10.12
RUN pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash && \
    pip install --no-cache-dir -U pip setuptools wheel

COPY --chown=user:user . .
# change dir since pip needs to seed whl folder
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt]
RUN python -m streamdiffusion.tools.install-tensorrt

COPY --from=frontend /app/dist ./frontend/dist

WORKDIR $HOME/app

USER user
CMD ["python", "main.py"]