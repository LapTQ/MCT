# MCT

start

```
$ sudo apt-get update
$ sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
$ sudo mkdir -p /etc/apt/keyrings
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
$ echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
$ sudo docker run hello-world
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2
$ sudo sudo systemctl restart docker
```


**SINGLE CAM TRACKING**




## Colab evaluation code

```
!git clone https://LapTQ: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@github.com/LapTQ/MCT.git
%cd MCT
%cd data
!wget https://motchallenge.net/data/MOT17.zip
!unzip MOT17.zip
%cd ../eval
!wget https://github.com/JonathonLuiten/TrackEval/archive/refs/heads/master.zip
!unzip master.zip
!mv TrackEval-master TrackEval
%cd TrackEval
!wget https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip
!unzip data.zip
%cd ../..
!pip install -r requirements.txt
```

```
%cd /content/MCT/
!python eval/predict_mot17.py
```

```
%cd /content/MCT/eval/TrackEval
!python scripts/run_mot_challenge.py --BENCHMARK MOT17 --TRACKERS_TO_EVAL SCT --METRICS HOTA --USE_PARALLEL False --NUM_PARALLEL_CORES 1
# !python scripts/run_mot_challenge.py --BENCHMARK MOT17 --TRACKERS_TO_EVAL SCT --METRICS CLEAR --USE_PARALLEL False --NUM_PARALLEL_CORES 1
```

```
%cd /content/MCT
!python3 mct/detection/detector.py
```

```
%cd /content/MCT
!python3 mct/tracking/tracker.py
```
