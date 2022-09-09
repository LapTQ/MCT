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


class DBBuilder(ABC):

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_product(self) -> DBBase:
        pass

    @abstractmethod
    def set_collection(self, collection) -> None:
        pass


class Pymongo(DBBase):

    def __init__(self):
        self.client = None
        self.database = None
        self.collection = None

    def update(self, tracklets: np.ndarray) -> None:
        """
        update tracks
        tracklets: [[frame, id, x1, y1, x2, y2, conf],...]
        """
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


class PymongoBuilder(DBBuilder):

    def __init__(self):
        self.mongo = None

    def reset(self) -> None:
        self.mongo = Pymongo()

    def get_product(self) -> DBBase:
        product = self.mongo
        self.reset()
        return product

    def set_client(self, host, port) -> None:
        self.mongo.client = MongoClient(host, port)

    def set_database(self, database) -> None:
        self.mongo.database = self.mongo.client[database]

    def set_collection(self, collection) -> None:
        self.mongo.collection = self.mongo.database[collection]


class MongoEngine(DBBase):

    def __init__(self):
        self.database = None
        self.detection_document = None
        self.track_document = None

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


class MongoEngineBuilder(DBBuilder):

    def __init__(self):
        self.mongo = None

    def reset(self) -> None:
        self.mongo = MongoEngine()

    def get_product(self) -> DBBase:
        product = self.mongo
        self.reset()
        return product

    def set_database(self, host, port, database) -> None:
        connect(host=f'mongodb://{host}:{port}/{database}')
        self.mongo.database = database

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

        self.mongo.detection_document = Detection
        self.mongo.track_document = Track


class DBDirector:

    def __init__(self):
        self._builder = None

    def set_builder(self, builder: DBBuilder) -> None:
        self._builder = builder

    def build_pymongo(self, host, port, database, collection) -> None:
        self._builder.reset()
        self._builder.set_client(host, port)
        self._builder.set_database(database)
        self._builder.set_collection(collection)

    def build_mongoengine(self, host, port, database, collection) -> None:
        self._builder.reset()
        self._builder.set_database(host, port, database)
        self._builder.set_collection(collection)


