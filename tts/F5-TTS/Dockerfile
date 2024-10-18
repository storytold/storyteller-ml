# Base CUDA image
FROM cnstark/pytorch:2.3.1-py3.10.15-cuda12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install necessary packages and git-lfs
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  tzdata \
  ffmpeg \
  parallel \
  git \
  nano \
  curl \
  software-properties-common \
  sudo \
  rsync && \
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
  apt-get install -y git-lfs && \
  git lfs install && \
  add-apt-repository ppa:deadsnakes/ppa && \
  rm -rf /var/lib/apt/lists/*

# Create directories for Python installation and model code
RUN mkdir -p /python_install /model_code/F5-TTS

# Set the working directory for Python installation
WORKDIR /python_install

# Copy the requirements.txt and install Python dependencies
COPY requirements*.txt /python_install/
# copies create real copies of python executable instead of symlinks, to make the virtualenv relocatable
RUN python3 -m venv --copies python && \
  python/bin/pip install --upgrade pip wheel setuptools && \
  python/bin/pip install --no-cache-dir -r requirements.txt -r requirements_eval.txt

WORKDIR /model_code/F5-TTS

# Copy the remaining contents to the model_code directory
COPY . /model_code/F5-TTS

# /model_code/F5-TTS/pretrained_models
ARG HF_DATASETS_CACHE="./pretrained_models"

# FIXME: this might not work without HF_TOKEN!!!
ENV HF_DATASETS_CACHE=$HF_DATASETS_CACHE

# initializes the model - downloads checkpoints
RUN  ["/bin/bash", "-c", "HF_DATASETS_CACHE=$HF_DATASETS_CACHE DOWNLOAD_ONLY=TRUE /python_install/python/bin/python inference-cli.py"]

ENTRYPOINT ["/bin/bash", "-c", ". /python_install/python/bin/activate && exec /bin/bash"]
