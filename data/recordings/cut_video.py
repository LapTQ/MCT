import cv2
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta

HERE = Path(__file__).parent

FILE_PATH = str(HERE / '2d_v4/videos/V_20230405_071605.mp4')
CAM_ID = 43
SEGMENTS = [[8975, 11060], [11552, 13640], [15066, 17614], [18363, 20643], [21321, 24120], [24869, 26994], [27649, 30169], [30472, 33035], [35057, 37624], [39711, 42391], [42907, 45287], [46023, 49215]]

OUT_DIR = str(HERE / '2d_v4/videos')
YEAR = 2023
MONTH = 4
DAY = 5
HOUR = 8
MINUTE = 30
SECOND = 0
MILISEC = 0


def main():
    cap = cv2.VideoCapture(FILE_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    day = DAY

    segment_count = 0
    for frame_count in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        
        ret, frame = cap.read()
        
        if not ret:
            break
            
        if SEGMENTS[segment_count][0] <= frame_count <= SEGMENTS[segment_count][1]:
        
            if frame_count == SEGMENTS[segment_count][0]:
                datetime_str = f"{YEAR}-{('00' + str(MONTH))[-2:]}-{('00' + str(day))[-2:]}_{('00' + str(HOUR))[-2:]}-{('00' + str(MINUTE))[-2:]}-{('00' + str(SECOND))[-2:]}-{('000000' + str(MILISEC))[-6:]}"
                OUT_PATH = f"{OUT_DIR}/{CAM_ID}_{('00000' + str(segment_count + 1))[-5:]}_{datetime_str}.avi"
                writer = cv2.VideoWriter(
                    OUT_PATH,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    fps,
                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                )
            
            segment_frame_count = frame_count - SEGMENTS[segment_count][0]
            datetime_ = datetime.strptime(datetime_str, '%Y-%m-%d_%H-%M-%S-%f') + timedelta(seconds=segment_frame_count/fps)
            cv2.putText(frame, f"frame: {str(segment_frame_count)}, datetime: {datetime_.strftime('%Y-%m-%d_%H-%M-%S-%f')}", (19, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), thickness=1)
            cv2.putText(frame, f"frame: {str(segment_frame_count)}, datetime: {datetime_.strftime('%Y-%m-%d_%H-%M-%S-%f')}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), thickness=1)
            writer.write(frame)
            
            
            if frame_count == SEGMENTS[segment_count][1]:
                writer.release()
                segment_count += 1
                day += 1
                
        if segment_count == len(SEGMENTS):
            break

    cap.release()
    
if __name__ == '__main__':
    main()
