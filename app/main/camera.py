import cv2


class Camera:

    def __init__(self) -> None:
        print('Camera')
        self.cap = cv2.VideoCapture(0)

    def get_frame(self):
        _, frame = self.cap.read()
        if _:
            return frame