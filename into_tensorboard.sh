#!/bin/bash
port=${1:-8880}
export MAIN_PATH="/share/generative-models"

nvidia-docker run -it \
      -w /root/packet-loss-concealment \
      -p $port:$port \
      -v $PWD:/root/object-detection-training \
      -v /share:/share \
      -m 4g \
      object-detect:latest /bin/bash -c "tensorboard --logdir $MAIN_PATH --port=$port"
