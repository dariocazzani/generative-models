#!/bin/bash
export MAIN_PATH="/share/generative-models"
export DATA_PATH="/share/generative-models/Data/"
sudo mkdir -p $DATA_PATH

echo "Downloading NSynth-test audio files if not present yet"
CURDIR=$PWD
cd $DATA_PATH
sudo wget -t 45 -nc http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz

# decompress and keep original files
if [ ! -d "$DATA_PATH/nsynth-test" ]; then
  sudo tar -zxvf nsynth-test.jsonwav.tar.gz
fi

port=${1:-8880}

xhost +
GPU=0 nvidia-docker run --privileged --rm -it \
  --env DISPLAY=$DISPLAY \
  --env="QT_X11_NO_MITSHM=1" \
  -e MAIN_PATH=$MAIN_PATH \
  -v /dev/video0:/dev/video0 \
  -v /dev/video1:/dev/video1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro  \
  -v $CURDIR:/root/generative-models \
  -p $port:$port \
  -v /share:/share \
  -w /root/generative-models \
  tensorflow:latest-gpu-py3 bash
