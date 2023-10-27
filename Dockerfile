FROM python:3.9.12

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
RUN pip install lap
RUN pip install ultralytics==8.0.20

# bash promt when container starts
ENTRYPOINT ["/bin/bash"]


