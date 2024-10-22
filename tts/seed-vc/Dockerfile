FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root
ENV SHELL=/bin/bash
ARG DEBIAN_FRONTEND=noninteractive

RUN set -x \
  && apt-get update \
  && apt-get -y install wget curl man git less openssl libssl-dev unzip build-essential tmux vim \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean

RUN mkdir -p /python_install /model_code/seed-vc

WORKDIR /python_install
COPY requirements.txt .

RUN python3 -m venv --copies python && \
  # python/bin/pip install --upgrade pip wheel setuptools && \
  python/bin/pip install --no-cache-dir -r requirements.txt

WORKDIR /model_code/seed-vc

COPY load_all_models.py hf_utils.py .

RUN ["/python_install/python/bin/python", "load_all_models.py"]

COPY . .

ENTRYPOINT ["/python_install/python/bin/python"]

