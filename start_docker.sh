port=${1:-8880}

nvidia-docker run --privileged --rm -it \
  -v $PWD:/root/generative-models \
  -p $port:$port \
  -v /share:/share \
  -w /root/generative-models \
  tensorflow:latest-gpu-py3 bash
