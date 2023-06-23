from flask_login import UserMixin
from flask import current_app
from werkzeug.security import generate_password_hash, check_password_hash
from app.extensions import db, login
import datetime
import json
import time
import numpy as np
from typing import Union, List, Dict
import cv2


class Scene:

    def __init__(
            self,
            width: Union[int, float, None] = None,
            height: Union[int, float, None] = None,
            roi: Union[np.ndarray, None] = None,
            roi_test_offset: Union[int, float] = 0,
            name='Scene'
    ) -> None:

        self.width = width
        self.height = height
        
        self.roi = roi   
        self._check_roi()
        self.roi_test_offset = roi_test_offset

        self.name = name
    

    def is_in_roi(self, x: Union[tuple, list, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if point(s) (x, y) is in the scene's roi.
        
        If x is not a numpy array represent only 1 point, then return bool object.
        """
        is_numpy = True
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            is_numpy = False
        
        x = np.float32(x.reshape(-1, 2))    # type: ignore
        ret = [cv2.pointPolygonTest(self.roi, p, True) >= self.roi_test_offset for p in x]      # type: ignore

        if not is_numpy and len(ret) == 1:
            return ret[0]
        else:
            return np.array(ret, dtype=bool)
        
    
    def has_roi(self) -> bool:
        return self.roi is not None
    

    def _check_roi(self):
        if self.roi is not None:
            assert isinstance(self.roi, np.ndarray)
            self.roi = np.int32(self.roi)   # type: ignore


class User(UserMixin, db.Model):
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(128), nullable=False)
    last_message_read_time = db.Column(db.DateTime)

    workshifts = db.relationship('RegisteredWorkshift', backref='user', lazy='dynamic')
    messages_received = db.relationship('Message', foreign_keys='Message.recipient_id', backref='recipient', lazy='dynamic')
    notifications = db.relationship('Notification', backref='user', lazy='dynamic')
    productivities = db.relationship('Productivity', backref='user', lazy='dynamic')
    regions = db.relationship('Region', primaryjoin='User.role == Region.role', lazy='dynamic')

    max_latency = datetime.timedelta(seconds=current_app.config['MAX_LATENCY'])
    max_absence = datetime.timedelta(seconds=current_app.config['MAX_ABSENCE'])
    

    __table_args__ = (
        db.CheckConstraint(role.in_(['manager', 'intern', 'engineer', 'admin'])),
    )


    def __repr__(self) -> str:
        return f'User(username={self.username}, role={self.role})'
    

    def __str__(self) -> str:
        return f'User(username={self.username}, role={self.role})'    


    def set_password(self, password):
        self.password_hash = generate_password_hash(password)


    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


    def new_messages(self):
        last_read_time = self.last_message_read_time or datetime.datetime(1900, 1, 1)
        return Message.query.filter_by(recipient=self).filter(
            Message.timestamp > last_read_time).count()


    def add_notification(self, name, data):
        self.notifications.filter_by(name=name).delete()
        n = Notification(name=name, payload_json=json.dumps(data), user=self)
        db.session.add(n)
        return n


    def overall_latency(self):     # TODO return queries
        c = 0
        now = datetime.datetime.now()
        for record in self.productivities.all():
            if record.arrival:
                latency = datetime.datetime.combine(datetime.date.today(), record.arrival) - \
                    datetime.datetime.combine(datetime.date.today(), record.dayshift.start_time)
                if latency <= datetime.timedelta(0):
                    latency = None
            elif now.time() < record.dayshift.start_time:
                latency = None
            else:
                latency = min(now, datetime.datetime.combine(datetime.date.today(), record.dayshift.end_time)) - \
                    datetime.datetime.combine(datetime.date.today(), record.dayshift.start_time)
            if latency:
                c += 1
        return c


    def overall_staying(self):
        num = datetime.timedelta(0)
        den = datetime.timedelta(0)
        now = datetime.datetime.now()
        for record in self.productivities.all():
            if now > datetime.datetime.combine(record.date, record.dayshift.end_time) and record.staying is not None:
                num += record.staying
                den += datetime.datetime.combine(datetime.date.today(), record.dayshift.end_time) - datetime.datetime.combine(datetime.date.today(), record.dayshift.start_time)
        
        if den == 0:
            return None
        
        return num/den


    def load_workareas(self):
        W = current_app.config['PIPELINE'].get('CAMERA_FRAME_WIDTH')
        H = current_app.config['PIPELINE'].get('CAMERA_FRAME_HEIGHT')
        assert W is not None
        assert H is not None
        self.scene_workareas = {}
        for r in self.regions.filter_by(type='workarea').all():
            cid = r.primary_cam_id
            if cid not in self.scene_workareas:
                self.scene_workareas[cid] = []
            roi = np.array(json.loads(r.points)) # type: ignore
            roi[:, 0] *= W
            roi[:, 1] *= H
            roi = roi.reshape(-1, 1, 2)
            scene = Scene(W, H, roi, current_app.config['PIPELINE'].get('ROI_TEST_OFFSET'))
            self.scene_workareas[cid].append(scene)

    
    def load_next_workshift(self):

        weekday_id_to_name = {
            0: 'Monday',
            1: 'Tuesday',
            2: 'Wednesday',
            3: 'Thursday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday'
        }
        now = datetime.datetime.now()
        today = now.date()
        upcomings = [(today + datetime.timedelta(days=i), DayShift.query.filter_by(name=dayshift_name).first()) 
                     for i in range(7) 
                     for dayshift_name in ['morning', 'afternoon']
        ]
        for i, ws in enumerate(upcomings):
            date, dayshift = ws
            weekday = date.weekday()
            workshift = self.workshifts.filter_by(
                day=weekday_id_to_name[weekday], 
                dayshift_id=dayshift.id
            ).first()
            if workshift is not None:
                if now < datetime.datetime.combine(date, dayshift.start_time):
                    self.next_workshift = workshift
                    print(f'next workshift: {self.next_workshift}')
                    break
        
        self.productivity_created = False
        self.arrived = False
        self.lateness_alert_sent = False
        self.n_misses = 0
        self.absence_alert_sent = False
        self.absence = datetime.timedelta(0)
        self.staying = self.next_workshift.dayshift.start_time - self.next_workshift.dayshift.end_time
        self.last_in_roi = self.next_workshift.dayshift.start_time


    def send_alert(self, message):
        managers = User.query.filter_by(role='manager').all()
        for manager in managers:
            msg = Message(recipient=manager, body=message)
            db.session.add(msg)
            manager.add_notification('unread_message_count', manager.new_messages())
        db.session.commit()        
    

    def update_detection(self, cid, dtime, loc):
        """update the detection record of the user

        This function assume that it is invoked as soon as the user is detected
        in the camera frame to ensure the real-time performance of the system.
        Unless, it might produce wrong productivity record and send false messages.
        
        Args:
            cid (int): camera id
            dtime (datetime.datetime): the time that the user is detected
            loc (None, array): the location of the user in the camera frame, in the format of (x, y)
        """

        assert hasattr(self, 'scene_workareas')
        assert dtime is not None
        assert datetime.datetime.now() - dtime < datetime.timedelta(seconds=current_app.config['PIPELINE'].get('DETECTION_EXPIRE_TIME'))
        assert hasattr(self, 'next_workshift')
        
        if isinstance(loc, np.ndarray):
            assert loc.shape == (2,)
            loc = tuple(loc)
        
        start_time = self.next_workshift.dayshift.start_time
        end_time = self.next_workshift.dayshift.end_time
        date = dtime.date()
        dayshift_id = self.next_workshift.dayshift_id

        condition_1 = not self.productivity_created
        condition_2 = dtime.time() + datetime.timedelta(minutes=30) > start_time
        condition_3 = dtime.time() <= start_time
        condition_4 = start_time < dtime < end_time

        # create productivity record 30 minutes before the workshift starts
        if condition_1 and condition_2: 
            self.current_productivity = Productivity(
                user_id=self.id, 
                date=date, 
                dayshift_id=dayshift_id
            )
            db.session.add(self.current_productivity)
            db.session.commit()
            self.productivity_created = True

        # during 30 minutes before the workshift starts
        if condition_2 and condition_3:
            if loc and not self.arrived:
                self.current_productivity.arrival = dtime.time()
                db.session.commit()
                self.arrived = True
        
        # during the workshift
        elif condition_4:   
    
            # if the user is detected
            if loc:
                # if the user has not arrived previously
                if not self.arrived:

                    # then save the arrival time
                    self.current_productivity.arrival = dtime.time()
                    db.session.commit()
                    self.arrived = True
                    
                    # and announce that the user has arrived
                    if self.lateness_alert_sent:
                        self.send_alert(f'User {self.username} arrived at {dtime}')
                        self.lateness_alert_sent = False
                
                # check if the user is in the workarea
                is_in_workarea = False
                for scene in self.scene_workareas.get(cid, []):
                    if scene.is_in_roi(loc):
                        is_in_workarea = True
                        break

                if is_in_workarea:
                    self.n_misses = 0
                    self.last_in_roi = dtime
                    self.staying -= self.absence
                    self.absence = datetime.timedelta(0)

                    if self.absence_alert_sent:
                        self.send_alert(f'User {self.username} is back at {dtime}')
                        self.absence_alert_sent = False
                else:
                    self.n_misses += 1
                    
            # if the user is not detected
            else:
                # if the user has not arrived
                if not self.arrived:
                    # then accumulate the latency and send alert if necessary
                    latency = dtime.time() - start_time
                    
                    if not self.lateness_alert_sent and latency > self.max_latency:
                        self.send_alert(f'User {self.username} was late for {latency}')
                        self.lateness_alert_sent = True
                
                # if the user has arrived previously
                else:
                    # then accumulate the absence time
                    self.n_misses += 1
            
            if self.arrived and self.n_misses > current_app.config['PIPELINE'].get('MAX_ABSENCE_FRAMES'):
                self.absence = dtime.time() - self.last_in_roi.time()
                if not self.absence_alert_sent and self.absence > self.max_absence:
                    self.send_alert(f'User {self.username} was absent for {self.absence}')
                    self.absence_alert_sent = True

        # after the workshift
        elif end_time < dtime:
            self.current_productivity.staying = self.staying
            db.session.commit()
            self.load_next_workshift()
    

@login.user_loader
def load_user(id):
    return User.query.get(int(id))


class DayShift(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(32))
    start_time = db.Column(db.Time)
    end_time = db.Column(db.Time)
    workshifts = db.relationship('RegisteredWorkshift', backref='dayshift', lazy='dynamic')
    productivities = db.relationship('Productivity', backref='dayshift', lazy='dynamic')

    __table_args__ = (
        db.CheckConstraint(name.in_(['morning', 'afternoon'])),
    )

    def __repr__(self):
        return f"DayShift(name={self.name}, start_time={self.start_time}, end_time={self.end_time})"


class RegisteredWorkshift(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    day = db.Column(db.String(28))
    dayshift_id = db.Column(db.Integer, db.ForeignKey('day_shift.id'))

    __table_args__ = (
        db.CheckConstraint(day.in_(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])),
    )

    def __repr__(self):
        username = db.session.get(User, self.user_id).username
        dayshift_name = db.session.get(DayShift, self.dayshift_id).name
        return f"RegisteredWorkshift(username={username}, day={self.day}, workshift_name={dayshift_name})"
    
    def __str__(self):
        return self.__repr__()
    

class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    num = db.Column(db.Integer, unique=True, nullable=False)
    address = db.Column(db.String(100), unique=True, nullable=False)

    def __repr__(self):
        return f'Camera(id={self.id}, num={self.num}, address={self.address})'


class Region(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(28))
    primary_cam_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)
    secondary_cam_id = db.Column(db.Integer, db.ForeignKey('camera.id'))
    points = db.Column(db.String(10000), nullable=False)
    role = db.Column(db.String(128), db.ForeignKey('user.role', name='fk_region_user_role'))

    __table_args__ = (
        db.CheckConstraint(type.in_(['checkin', 'overlap', 'workarea'])),
    )

    def __repr__(self):
        primary_cam_num = db.session.get(Camera, self.primary_cam_id).num
        if self.secondary_cam_id is None:
            secondary_cam_num = None    
        else:
            secondary_cam_num = db.session.get(Camera, self.secondary_cam_id).num
        return f'Region(type={self.type}, primary_cam_num={primary_cam_num}, secondary_cam_num={secondary_cam_num}, role={self.role}, points=...)'


class CameraMatchingPoint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    primary_cam_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)      # sourse
    secondary_cam_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)    # destination
    points = db.Column(db.String(10000), nullable=False)                                    # source, destination

    def __repr__(self):
        primary_cam_num = db.session.get(Camera, self.primary_cam_id).num
        secondary_cam_num = db.session.get(Camera, self.secondary_cam_id).num
        return f'CameraMatchingPoint(primary_cam_num={primary_cam_num}, secondary_cam_id={secondary_cam_num}, points=...)'


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recipient_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    body = db.Column(db.String(140))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.datetime.now)


class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    timestamp = db.Column(db.Float, index=True, default=time.time())
    payload_json = db.Column(db.Text)

    def get_data(self):
        return json.loads(str(self.payload_json))
    

class Productivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    date = db.Column(db.Date)
    dayshift_id = db.Column(db.Integer, db.ForeignKey('day_shift.id'))
    arrival = db.Column(db.Time)
    staying = db.Column(db.Interval)

    def __repr__(self):
        return f'Productivity(user_id={self.user_id} \tdate={self.date} \tdayshift_id={self.dayshift_id} \tarrival={self.arrival} \tstaying={self.staying}'

    # >>> from datetime import timedelta, time, datetime
    # >>> p = Productivity(user_id=3, date=datetime(day=5, month=6, year=2023).date(), dayshift_id=1, arrival=time(hour=8, minute=29, second=10), staying=timedelta(hours=2, minutes=10))


class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    cam_id = db.Column(db.Integer, db.ForeignKey('camera.id'))
    date = db.Column(db.Date, db.ForeignKey('productivity.date'))
    frame_time = db.Column(db.Time)
    frame_id = db.Column(db.Integer)
    track_id = db.Column(db.Integer)
    detection_mode = db.Column(db.String(140))
    tracking_mode = db.Column(db.String(140))
    det = db.Column(db.String(10000))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


class STA(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    det_id_1 = db.Column(db.Integer, db.ForeignKey('detection.id'))
    det_id_2 = db.Column(db.Integer, db.ForeignKey('detection.id'))
    loc_infer_mode = db.Column(db.Integer)