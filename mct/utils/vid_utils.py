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
    def read(self) -> Tuple[bool, np.ndarray]:
        """
        return ret, frame
        """
        pass

    @abstractmethod
    def release(self) -> None:
        pass


class LoaderBuilder(ABC):

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_product(self) -> None:
        pass

    @abstractmethod
    def set_input(self, path) -> None:
        pass


class VideoLoader(LoaderBase):

    def __init__(self):
        self.pool = None
        self.length = None

    def get_fps(self) -> float:
        return self.pool.get(cv2.CAP_PROP_FPS)

    def __len__(self) -> int:
        return self.length

    def read(self) -> Tuple[bool, np.ndarray]:
        ret, frame = self.pool.read()
        return ret, frame

    def release(self) -> None:
        self.pool.release()


class VideoLoaderBuilder(LoaderBuilder):

    def __init__(self):
        self.loader = None

    def reset(self) -> None:
        self.loader = VideoLoader()

    def get_product(self) -> None:
        product = self.loader
        self.reset()
        return product

    def set_input(self, path) -> None:
        print(path)
        self.loader.pool = cv2.VideoCapture(path)
        if path == 0:
            self.loader.length = int(1e9)
        else:
            self.loader.length = int(self.loader.pool.get(cv2.CAP_PROP_FRAME_COUNT))


class ImageFolderLoader(LoaderBase):

    def __init__(self):
        self.pool = None
        self.fps = None
        self.length = None

        self.i = 0

    def get_fps(self) -> float:
        return self.fps

    def __len__(self) -> int:
        return self.length

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


class ImageFolderLoaderBuilder(LoaderBuilder):

    def __init__(self):
        self.loader = None

    def reset(self) -> None:
        self.loader = ImageFolderLoader()

    def get_product(self) -> None:
        product = self.loader
        self.reset()
        return product

    def set_input(self, path) -> None:
        self.loader.pool = [os.path.join(path, filename) for filename in sorted(os.listdir(path))]

    def set_metadata(self, path) -> None:
        if path is None:
            self.loader.fps = 30
            self.loader.length = len(self.loader.pool)
        else:
            cfg = ConfigParser()
            cfg.read(path)
            self.loader.fps = float(cfg['Sequence']['frameRate'])
            self.loader.length = int(cfg['Sequence']['seqLength'])


class LoaderDirector:

    def __init__(self):
        self._builder = None

    def set_builder(self, builder):
        self._builder = builder

    def build_videoloader(self, path):
        self._builder.reset()
        self._builder.set_input(path)

    def build_imagefolderloader(self, imdir_path, ini_path):
        self._builder.reset()
        self._builder.set_input(imdir_path)
        self._builder.set_metadata(ini_path)




