# Base CUDA image
FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime

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
  software-properties-common && \
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

# initializes the model - downloads checkpoints
# FIXME: is isn't working
# ARG HF_DATASETS_CACHE="./pretrained_models"
# ENV HF_DATASETS_CACHE=$HF_DATASETS_CACHE
# RUN  ["/bin/bash", "-c", "HF_DATASETS_CACHE=$HF_DATASETS_CACHE DOWNLOAD_ONLY=TRUE /python_install/python/bin/python inference-cli.py"]

# checkpoints at the default location ~/.cache/huggingface/hub
RUN ["/bin/bash", "-c", "DOWNLOAD_ONLY=TRUE /python_install/python/bin/python inference-cli.py"]

ENTRYPOINT ["/bin/bash", "-c", ". /python_install/python/bin/activate && exec /bin/bash"]
