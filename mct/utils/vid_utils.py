from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np
import os
from configparser import ConfigParser

class LoaderBase(ABC):

    @abstractmethod
    def get_fps(self) -> float:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def get_height(self) -> int:
        pass

    @abstractmethod
    def get_width(self) -> int:
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, np.ndarray]:
        """return ret, frame"""
        pass

    @abstractmethod
    def release(self) -> None:
        pass


class BuilderBase(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _reset(self) -> None:
        pass

    @abstractmethod
    def get_product(self) -> LoaderBase:
        pass


class VideoLoader(LoaderBase):

    class Builder(BuilderBase):

        def __init__(self, path):
            self._reset()

            cap = cv2.VideoCapture(path)
            self._product.pool = cap
            self._product.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._product.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._product.fps = cap.get(cv2.CAP_PROP_FPS)
            if path == 0:
                self._product.length = int(1e9)
            else:
                self._product.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        def _reset(self) -> None:
            self._product = VideoLoader()

        def get_product(self) -> LoaderBase:
            product = self._product
            self._reset()
            return product

    def get_fps(self) -> float:
        return self.fps

    def __len__(self) -> int:
        return self.length

    def get_height(self) -> int:
        return self.height

    def get_width(self) -> int:
        return self.width

    def read(self) -> Tuple[bool, np.ndarray]:
        ret, frame = self.pool.read()
        return ret, frame

    def release(self) -> None:
        self.pool.release()


class ImageFolderLoader(LoaderBase):

    class Builder(BuilderBase):

        def __init__(self, path, meta):
            self._reset()

            cap = [os.path.join(path, filename) for filename in sorted(os.listdir(path))]
            self._product.pool = cap
            self._product.i = 0
            if meta is None:
                self._product.fps = 30
                self._product.length = len(cap)
                H, W = cv2.imread(cap[0]).shape[:2]
                self._product.height = H
                self._product.width = W
            else:
                cfg = ConfigParser()
                cfg.read(meta)
                self._product.fps = float(cfg['Sequence']['frameRate'])
                self._product.length = int(cfg['Sequence']['seqLength'])
                self._product.height = int(cfg['Sequence']['imHeight'])
                self._product.width = int(cfg['Sequence']['imWidth'])


        def _reset(self) -> None:
            self._product = ImageFolderLoader()

        def get_product(self) -> LoaderBase:
            product = self._product
            self._reset()
            return product

    def get_fps(self) -> float:
        return self.fps

    def __len__(self) -> int:
        return self.length

    def get_height(self) -> int:
        return self.height

    def get_width(self) -> int:
        return self.width

    def read(self) -> Tuple[bool, np.ndarray]:
        if self.i < len(self.pool):
            frame = cv2.imread(self.pool[self.i])
            self.i += 1
            ret = True
        else:
            frame = None
            ret = False
        return ret, frame

    def release(self) -> None:
        pass


