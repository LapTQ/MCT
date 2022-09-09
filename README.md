# MCT


**SINGLE CAM TRACKING**

```
sudo docker run -d -p 8000:27017 --name laptq_mongodb mongo
```

Yếu tố ảnh hưởng đến SORT:
* Chất lượng của detection: YOLOv5 mosiac augment mặc định sinh ra nhiều FP
* Kích thước ảnh đầu vào của YOLOv5: lớn hơn chưa chắc tốt hơn, có thể làm trầm trọng hiện tượng id switch nếu 1 người "bị" YOLO detect 1 phần cơ thể ngay sau khi bị osculated => IOU không đủ lớn
* các thông số của kalman filter
* FPS



