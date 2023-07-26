FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04

WORKDIR /home

RUN apt-get update -y
RUN apt-get install -y python-is-python3
RUN apt-get -y install python3-pip
RUN apt-get install nano -y

# <src> is relative to context in docker-compose.yml
# <des> is relative to WORKDIR
COPY ./requirements.txt requirements.txt

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get install ffmpeg libsm6 libxext6 libgtk2.0-dev pkg-config -y

RUN pip install -r requirements.txt

# bash promt when container starts
ENTRYPOINT ["/bin/bash"]


