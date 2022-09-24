from abc import ABC, abstractmethod

import numpy as np

from pymongo import MongoClient
from mongoengine import connect, disconnect, Document, IntField, FloatField, EmbeddedDocument, ListField, EmbeddedDocumentListField



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


class BuilderBase(ABC):

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
        """tracklets: [[frame, id, x1, y1, x2, y2, conf],...]"""
        assert len(tracklets.shape) == 2 and tracklets.shape[1] == 7, 'Invalid tracklets shape'
        for tl in tracklets:
            self.collection.update_one(
                {'trackid': int(tl[1])},
                {'$push': {'detections': {'frameid': int(tl[0]),
                                          'box': tl[2:6].tolist(),  # xyxy
                                          'score': tl[6]
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
                frameid = IntField(required=True)
                box = ListField(field=FloatField(), default=[], required=True)
                score = FloatField(required=True)

            class Track(Document):
                trackid = IntField(db_field='trackid', required=True, unique=True)
                detections = EmbeddedDocumentListField(Detection, db_field='detections', default=[], required=True)
                meta = {
                    'collection': collection,
                    # 'indexes': ['trackid']  # TODO check
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
        tracklets: [[frame, id, x1, y1, x2, y2, conf],...]
        """
        for tl in tracklets:
            rec_detection = self.detection_document(
                frameid=int(tl[0]),
                box=tl[2:6].tolist(),  # xyxy
                score=tl[6]
            )
            self.track_document.objects(trackid=int(tl[1])).update(push__detections=rec_detection, upsert=True)

    def close(self) -> None:
        disconnect(self.database)



