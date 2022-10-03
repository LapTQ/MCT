# MCT

**SINGLE CAM TRACKING**



Yếu tố ảnh hưởng đến SORT:
* Chất lượng của detection: YOLOv5 mosiac augment mặc định sinh ra nhiều FP
* Kích thước ảnh đầu vào của YOLOv5: lớn hơn chưa chắc tốt hơn, có thể làm trầm trọng hiện tượng id switch nếu 1 người "bị" YOLO detect 1 phần cơ thể ngay sau khi bị osculated => IOU không đủ lớn
* các thông số của kalman filter
* FPS


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
