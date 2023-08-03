# Multi-Camera Tracking for Employee Behavior Monitoring

This the code implementation for my thesis work at Bach Khoa Ha Noi University - Vietnam. The thesis focus on 3 main objectives:
* **Research**: Develop a Spatio-Temporal Association for MCT that can work well when people have similar appearance, which is a challenge for popular visual-based Re-ID methods.
* **Technology**: Master Computer Vision and Deep Learning frameworks and libraries. This project have also employed and modified in the source code of YOLOv5, YOLOv7, YOLOv8, SORT, ByteTrack, TrackEval.
* **Application**: Develop an software system using Flask that showcases the applicability of the proposed method.

## Setup

Clone the repository:

```bash
git clone https://github.com/LapTQ/MCT.git
cd MCT
```

Download YOLOv7 pose weight from [here](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) and put it in `mct/weights/`.

Then, download videos from this Telegram [group](https://t.me/laptq_file_storage/4) and put them to `data/recordings/2d_v4/videos/`.

## Install dependencies

### Option 1: Docker

```bash
docker compose up
```

***Note***: You might need to [upgrade](https://docs.docker.com/engine/install/ubuntu/) your docker engine to the lastest version.

### Option 2: Conda

```bash
conda create --name MCT python==3.9.12
conda activate MCT
pip install -r requirements.txt
pip install lap
```

## Start application

If you're using docker, start the container:

```bash
docker start laptq_mct_backend -d
docker attach laptq_mct_backend
cd MCT
```

Else if you're using conda, activate the environment:

```bash
conda activate MCT
```

Then, run the application:

```bash
flask run
```

You can access the application via http://0.0.0.0:5555 from your browser.

## Reports and demo

Detailed thesis report and video demo can be found [here](docs/reports)
