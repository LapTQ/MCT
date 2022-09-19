# MCT

**SINGLE CAM TRACKING**



Yếu tố ảnh hưởng đến SORT:
* Chất lượng của detection: YOLOv5 mosiac augment mặc định sinh ra nhiều FP
* Kích thước ảnh đầu vào của YOLOv5: lớn hơn chưa chắc tốt hơn, có thể làm trầm trọng hiện tượng id switch nếu 1 người "bị" YOLO detect 1 phần cơ thể ngay sau khi bị osculated => IOU không đủ lớn
* các thông số của kalman filter
* FPS

```
HOTA: SCT-pedestrian               HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA      RHOTA     HOTA(0)   LocA(0)   HOTALocA(0)
COMBINED                           31.872    25.28     40.647    26.995    71.427    43.371    82.584    81.938    33.005    40.478    75.301    30.48

CLEAR: SCT-pedestrian              MOTA      MOTP      MODA      CLR_Re    CLR_Pr    MTR       PTR       MLR       sMOTA     CLR_TP    CLR_FN    CLR_FP    IDSW      MT        PT        ML        Frag
COMBINED                           25.585    79.799    26.029    31.911    84.436    15.629    36.142    48.23     19.138    107505    229386    19817     1496      256       592       790       1928

Count: SCT-pedestrian              Dets      GT_Dets   IDs       GT_IDs    
COMBINED                           127322    336891    2891      1638
```


