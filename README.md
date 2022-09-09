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

```
HOTA: SCT-pedestrian               HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA      RHOTA     HOTA(0)   LocA(0)   HOTALocA(0)
COMBINED                           25.251    17.304    37.027    17.852    76.015    39.493    83.134    82.714    25.671    31.458    78.211    24.604    

Count: SCT-pedestrian              Dets      GT_Dets   IDs       GT_IDs    
COMBINED                           79117     336891    1729      1638      
```



