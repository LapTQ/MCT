import cv2
from pathlib import Path
import cv2
from tqdm import tqdm

HERE = Path(__file__).parent

FILE_PATH = str(HERE / '2d_v4/videos/IMG_8496.MOV')
OUT_PATH = str(HERE / '2d_v4/videos/IMG_8496_put_frame_count.MOV')


def main():
    cap = cv2.VideoCapture(FILE_PATH)
    writer = cv2.VideoWriter(
        OUT_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )
    
    milisec = 0
    for frame_count in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        
        if not ret:
            break
            
        cv2.putText(frame, f"frame: {frame_count}, milisec: {milisec}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
        writer.write(frame)
        
        milisec += 1e3 / cap.get(cv2.CAP_PROP_FPS)

    cap.release()
    writer.release()
    
    
if __name__ == '__main__':
    main()
