FROM gcr.io/tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
		git \
		wget \
		python3-tk \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
