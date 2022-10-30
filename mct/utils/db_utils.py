from abc import ABC, abstractmethod
from threading import Thread, Lock

import numpy as np
from datetime import datetime

from pymongo import MongoClient
from mongoengine import connect, disconnect, Document, IntField, FloatField, EmbeddedDocument, ListField, EmbeddedDocumentListField, DateTimeField


class DBBase(ABC):

    @abstractmethod
    def update(self, tracklets: np.ndarray) -> None:
        """
        update tracks
        tracklets: [[frame, id, x1, y1, x2, y2, conf],...]
        """
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class SingletonMeta(type):

    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class BuilderBase(metaclass=SingletonMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _reset(self) -> None:
        pass

    @abstractmethod
    def get_product(self) -> DBBase:
        pass


class Pymongo(DBBase):

    class Builder(BuilderBase):

        def __init__(self, host, port):
            self._reset()

            self._product.client = MongoClient(host, port)

        def set_database(self, database):
            self._product.database = self._product.client[database]

            return self

        def set_collection(self, collection):
            self._product.collection = self._product.database[collection]

            return self


        def _reset(self) -> None:
            self._product = Pymongo()

        def get_product(self) -> DBBase:
            product = self._product
            self._reset()
            return product

    def update(self, tracklets: np.ndarray) -> None:
        """tracklets: [[cam_id, videoid, time, frame, id, x1, y1, x2, y2, conf],...]"""
        assert len(tracklets.shape) == 2 and tracklets.shape[1] == 10, 'Invalid tracklets shape'
        for tl in tracklets:
            self.collection.update_one(
                {'camid': int(tl[0]), 'videoid': int(tl[1]), 'trackid': int(tl[4])},
                {'$push': {'detections': {'time': datetime.fromtimestamp(tl[2]),
                                          'frameid': int(tl[3]),
                                          'box': tl[5:9].tolist(),  # xyxy
                                          'score': tl[9]
                                          }
                           }
                 },
                upsert=True
            )

    def close(self) -> None:
        self.client.close()


class MongoEngine(DBBase):

    class Builder(BuilderBase):

        def __init__(self, host, port):
            self._reset()

            self.host = host
            self.port = port

        def set_databse(self, database):
            connect(host=f'mongodb://{self.host}:{self.port}/{database}')
            self._product.database = database

            return self

        def set_collection(self, collection) -> None:
            class Detection(EmbeddedDocument):
                time = DateTimeField(default=None)
                frameid = IntField(required=True)
                box = ListField(field=FloatField(), default=[], required=True)
                score = FloatField(required=True)

            class Track(Document):
                camid = IntField(db_field='camid', default=None)
                videoid = IntField(db_field='videoid', default=None)
                trackid = IntField(db_field='trackid', required=True)
                detections = EmbeddedDocumentListField(Detection, db_field='detections', default=[], required=True)
                meta = {
                    'collection': collection,
                    'indexes': [{'fields': ('camid', 'videoid', 'trackid'),
                                 'unique': True}]  # TODO check
                }

            self._product.detection_document = Detection
            self._product.track_document = Track

            return self

        def _reset(self) -> None:
            self._product = MongoEngine()

        def get_product(self) -> DBBase:
            product = self._product
            self._reset()
            return product

    def update(self, tracklets: np.ndarray) -> None:
        """
        update tracks
        tracklets: [[camid, videoid, time, frame, id, x1, y1, x2, y2, conf],...]
        """
        for tl in tracklets:
            rec_detection = self.detection_document(
                time=datetime.fromtimestamp(tl[2]),
                frameid=int(tl[3]),
                box=tl[5:9].tolist(),  # xyxy
                score=tl[9]
            )
            self.track_document.objects(
                camid=int(tl[0]),
                videoid=int(tl[1]),
                trackid=int(tl[4])
            ).update(push__detections=rec_detection, upsert=True)

    def close(self) -> None:
        disconnect(self.database)



