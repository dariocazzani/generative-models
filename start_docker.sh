export MAIN_PATH="/share/generative-models"
port=${1:-8880}

xhost +
GPU=0 nvidia-docker run --privileged --rm -it \
  --env DISPLAY=$DISPLAY \
  --env="QT_X11_NO_MITSHM=1" \
  -e MAIN_PATH=$MAIN_PATH \
  -v /dev/video0:/dev/video0 \
  -v /dev/video1:/dev/video1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro  \
  -v $PWD:/root/generative-models \
  -p $port:$port \
  -v /share:/share \
  -w /root/generative-models \
  tensorflow:latest-gpu-py3 bash
