FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y install git ffmpeg libsm6 libxext6
RUN pip install ipython ipykernel matplotlib tensorboard scikit-image configargparse opencv-python