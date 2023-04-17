
CAM1_FPS = 30.0
CAM1_START = 8115
"""
CAM2_FPS = 30.011
CAM2_START = 8293   # corresponds to CAM1_START
END_OFFSET = -30
"""
#"""
CAM2_FPS = 29.903
CAM2_START = 8975   # corresponds to CAM1_START
END_OFFSET = -90
#"""


CAM1_SEGMENTS = [
    [8115, 10298],
    [10701, 12886],
    [14226, 16873],
    [17534, 19912],
    [20502, 23400],
    [24061, 26283],
    [26850, 29469],
    [29682, 32344],
    [34282, 36948],
    [38951, 41730],
    [42158, 44636],
    [45284, 48576]
]

CAM2_SEGMENTS = []
for cam1_start, cam1_end in CAM1_SEGMENTS:
    cam2_start = int((cam1_start - CAM1_START) / CAM1_FPS * CAM2_FPS + CAM2_START)
    cam2_end = int((cam1_end - CAM1_START) / CAM1_FPS * CAM2_FPS + CAM2_START + END_OFFSET)
    CAM2_SEGMENTS.append([cam2_start, cam2_end])
    
print(CAM2_SEGMENTS)
