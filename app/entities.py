from flask_login import UserMixin
from flask import current_app
from werkzeug.security import generate_password_hash, check_password_hash
from app.extensions import db, login, fake_clock, moment
import datetime
import json
import numpy as np
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from mct.sta.engines import Scene


logger = logging.getLogger(__name__)

WEEKDAY_ID_TO_NAME = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}


class User(UserMixin, db.Model):
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    name = db.Column(db.String(64))
    phone = db.Column(db.String(64))
    email = db.Column(db.String(64))
    address = db.Column(db.String(64))
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(128), nullable=False)
    last_message_read_time = db.Column(db.DateTime, default=datetime.datetime(2001, 5, 24))

    workshifts = db.relationship('RegisteredWorkshift', backref='user', lazy='dynamic')
    messages_received = db.relationship('Message', foreign_keys='Message.recipient_id', backref='recipient', lazy='dynamic')
    notifications = db.relationship('Notification', backref='user', lazy='dynamic')
    productivities = db.relationship('Productivity', backref='user', lazy='dynamic')
    regions = db.relationship('Region', primaryjoin='User.role == Region.role', lazy='dynamic')
    

    __table_args__ = (
        db.CheckConstraint(role.in_(['manager', 'intern', 'engineer', 'admin'])),
    )


    def __repr__(self) -> str:
        return  f'User(username={self.username}, role={self.role_name}, last_message_read_time={self.last_message_read_time})'    
    

    def __str__(self) -> str:
        return f'{self.role_name} {self.name}'


    def set_password(self, password):
        self.password_hash = generate_password_hash(password)


    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    

    @property
    def role_name(self):
        if self.role == 'intern':
            return 'sale associate'
        elif self.role == 'engineer':
            return 'warehouse staff'
        else:
            return self.role


    def new_messages(self):
        return Message.query.filter_by(recipient=self).filter(
            Message.timestamp > self.last_message_read_time).count()


    def add_notification(self, name, data):
        self.notifications.filter_by(name=name).delete()
        n = Notification(
            name=name, 
            payload_json=json.dumps(data), 
            user=self,
            timestamp=fake_clock.now().timestamp()
        )
        db.session.add(n)
        return n


    def overall_latency(self):     # TODO return queries
        c = 0
        for r in self.productivities.all():
            prod_info = self.extract_productivity(r)
            if prod_info['latency']:
                c += 1
        return c


    def overall_staying(self):
        num = datetime.timedelta(0)
        den = datetime.timedelta(0)

        for r in self.productivities.all():
            prod_info = self.extract_productivity(r)
            if prod_info['staying']:
                num += prod_info['staying']
                den += prod_info['shift_duration']
        
        if den == datetime.timedelta(0):
            return None
        
        return num/den
    

    def extract_productivity(self, record):
        now = fake_clock.now()
        start_time = datetime.datetime.combine(record.date, record.dayshift.start_time)
        end_time = datetime.datetime.combine(record.date, record.dayshift.end_time)

        arrival = record.arrival
        if now < start_time:
            latency = None
        else:
            if record.arrival is not None:
                arrival = record.arrival.replace(microsecond=0)

                latency = datetime.datetime.combine(record.date, arrival) - start_time
                if latency <= datetime.timedelta(0):
                    latency = None
            else:
                latency = now - start_time

        if now > start_time and record.staying is not None:
            staying = datetime.timedelta(seconds=round(record.staying.total_seconds()))
            staying_percent = record.staying / (end_time - start_time)
            
        else:
            staying = None
            staying_percent = None

        return {
            'day': WEEKDAY_ID_TO_NAME[record.date.weekday()],
            'date': record.date,
            'dayshift': record.dayshift,
            'arrival': arrival,
            'latency': latency,
            'staying': staying,
            'shift_duration': end_time - start_time,
            'staying_percent': staying_percent
        }


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

        logger.info('Loaded workareas for user %s', self.username)

    
    def load_next_workshift(self):

        if not hasattr(self, 'max_latency'):
            self.max_latency = datetime.timedelta(seconds=current_app.config['MAX_LATENCY'])

        if not hasattr(self, 'max_absence'):
            self.max_absence = datetime.timedelta(seconds=current_app.config['MAX_ABSENCE'])

        now = fake_clock.now()
        today = now.date()
        upcomings = [(today + datetime.timedelta(days=i), DayShift.query.filter_by(name=dayshift_name).first()) 
                     for i in range(7) 
                     for dayshift_name in ['morning', 'afternoon']
        ]
        self.has_workshifts = False
        for i, ws in enumerate(upcomings):
            date, dayshift = ws
            weekday = date.weekday()
            workshift = self.workshifts.filter_by(
                day=WEEKDAY_ID_TO_NAME[weekday], 
                dayshift_id=dayshift.id
            ).first()
            if workshift is not None:
                if now < datetime.datetime.combine(date, dayshift.start_time):
                    self.has_workshifts = True
                    break
        
        if not self.has_workshifts:
            return
        
        self._productivity_created = False
        self._arrived = False
        self._lateness_alert_sent = False
        self._n_misses = 0
        self._absence_alert_sent = False
        self._absence = datetime.timedelta(0)
        self._staying = datetime.datetime.combine(today, workshift.dayshift.end_time) - \
             datetime.datetime.combine(today, workshift.dayshift.start_time)
        self._last_in_roi = datetime.datetime.combine(today, workshift.dayshift.start_time)

        self._ws_start_time = datetime.datetime.combine(date, workshift.dayshift.start_time)
        self._ws_end_time = datetime.datetime.combine(date, workshift.dayshift.end_time)
        self._ws_dayshift_id = workshift.dayshift_id

        logger.info('Loaded next workshift for %s: %s %s %s', self.name, workshift.day, date, workshift.dayshift.name)


    def send_alert(self, message):
        managers = User.query.filter_by(role='manager').all()
        for manager in managers:
            msg = Message(recipient=manager, body=message)
            db.session.add(msg)
            manager.add_notification('unread_message_count', manager.new_messages())
        db.session.commit()

        logger.warning('Message managers: %s', message)   


    def _check_in_workarea(self, cid, loc):
        is_in_workarea = False
        for scene in self.scene_workareas.get(cid, []):
            if scene.is_in_roi(loc):
                is_in_workarea = True
                break
        return is_in_workarea
    

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

        if not self.has_workshifts:
            return
        
        if dtime == '<EOS>':
            if self._productivity_created:
                self.current_productivity.staying = self._staying
                db.session.commit()
                logger.info('Updated staying in %s', self.current_productivity)
            return
                
        assert dtime is not None
        dtime = datetime.datetime.fromtimestamp(dtime)
        # assert fake_clock.now() - dtime < datetime.timedelta(seconds=current_app.config['DETECTION_EXPIRE_TIME']), 'The annotation comes slower than frame, please optimize the pipeline'
        assert hasattr(self, 'scene_workareas')
        
        if isinstance(loc, np.ndarray):
            assert loc.shape == (2,)
            loc = tuple(loc)
        
        date = dtime.date()
        
        condition_1 = not self._productivity_created
        condition_2 = dtime + datetime.timedelta(minutes=30) > self._ws_start_time
        condition_3 = dtime <= self._ws_start_time
        condition_4 = self._ws_start_time < dtime < self._ws_end_time

        # create productivity record 30 minutes before the workshift starts
        if condition_1 and condition_2: 
            self.current_productivity = Productivity(
                user_id=self.id, 
                date=date, 
                dayshift_id=self._ws_dayshift_id
            )
            db.session.add(self.current_productivity)
            db.session.commit()
            self._productivity_created = True
            logger.info('Create %s', self.current_productivity)

        # during 30 minutes before the workshift starts
        if condition_2 and condition_3:
            if loc and not self._arrived:
                self.current_productivity.arrival = dtime.time()
                db.session.commit()
                self._arrived = True
                logger.info('%s arrived in time at %s', self, dtime.time()) # .replace(microsecond=0)
        
        # during the workshift
        elif condition_4:   
    
            # if the user is detected
            if loc:
                # if the user has not arrived previously
                if not self._arrived:

                    # then save the arrival time
                    self.current_productivity.arrival = dtime.time()
                    db.session.commit()
                    self._arrived = True
                    self._last_in_roi = dtime
                    
                    # and announce that the user has arrived
                    if self._lateness_alert_sent:
                        self.send_alert(f'{self.name} arrived late at {dtime.time().replace(microsecond=0)}') # .replace(microsecond=0)
                        self._lateness_alert_sent = False

                if self._check_in_workarea(cid, loc):
                    self._n_misses = 0
                    self._last_in_roi = dtime
                    self._absence = datetime.timedelta(0)

                    if self._absence_alert_sent:
                        self.send_alert(f'{self.name} is back to work area at {dtime.time().replace(microsecond=0)}') # .replace(microsecond=0)
                        self._absence_alert_sent = False
                else:
                    self._n_misses += 1
                    
            # if the user is not detected
            else:
                # if the user has not arrived
                if not self._arrived:
                    # then send alert if necessary
                    latency = dtime - self._ws_start_time
                    if not self._lateness_alert_sent and latency > self.max_latency:
                        self.send_alert(f'{self.name} was late for {datetime.timedelta(seconds=round(latency.total_seconds()))}') # datetime.timedelta(seconds=round(latency.total_seconds()))
                        self._lateness_alert_sent = True
                
                self._n_misses += 1      # accumulate the absence time
            
            # if the user does not stay in work area for a certain period
            if  self._n_misses > current_app.config['MAX_ABSENCE_FRAMES']:
                _absence = dtime - self._last_in_roi
                self._staying -= _absence - self._absence
                self._absence = _absence
                if self._arrived and not self._absence_alert_sent and self._absence > self.max_absence:
                    self.send_alert(f'{self.name} was absent from work area since {self._last_in_roi.time().replace(microsecond=0)}') # .replace(microsecond=0)
                    self._absence_alert_sent = True

        # after the workshift
        elif self._ws_end_time < dtime:
            if self._productivity_created:
                self.current_productivity.staying = self._staying
                db.session.commit()
                logger.info('Updated staying in %s', self.current_productivity)
                self.load_next_workshift()
    

    def update_hint(self, cam_id, track_id):

        if not hasattr(self, '_hint_cam_id') or self._hint_cam_id is None:
            self._hint_cam_id = cam_id
        
        if not hasattr(self, '_hint_track_id') or self._hint_track_id is None:
            self._hint_track_id = track_id

        if not hasattr(self, '_hint_age'):
            self._hint_age = -1
        
        if not hasattr(self, '_hint_history'):
            self._hint_history = []

        if cam_id is None or track_id is None:
            if self._hint_age >= 0:
                self._hint_age += 1
        else:
            self._hint_history.append((cam_id, track_id))
            self._hint_age = 0
        
        if self._hint_age > current_app.config['SWITCH_CAM_MAX_AGE']:
            if len(self._hint_history) > 0:
                freq = {}
                for i, m in enumerate(self._hint_history):
                    freq[m] = freq.get(m, 0) + i                # more weight on newer candidate
                self._hint_cam_id, self._hint_track_id = max(freq, key=freq.get)    # type: ignore
                self._hint_age = 0
                self._hint_history = []

    
    def get_hint(self):
        if not hasattr(self, '_hint_cam_id'):
            self._hint_cam_id = None
        
        if not hasattr(self, '_hint_track_id'):
            self._hint_track_id = None
        
        return self._hint_cam_id, self._hint_track_id

    

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
    
    def __str__(self):
        return f"{self.name} (from {self.start_time} to {self.end_time})"


class RegisteredWorkshift(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    day = db.Column(db.String(28))
    dayshift_id = db.Column(db.Integer, db.ForeignKey('day_shift.id'))

    __table_args__ = (
        db.CheckConstraint(day.in_(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])),
    )

    def __repr__(self):
        return f"RegisteredWorkshift(user_id={self.user_id}, day={self.day}, dayshift_id={self.dayshift_id})"
    
    def __str__(self):
        return f"workshift on {self.day} {self.dayshift.name} for user {self.user.name}"
    

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
    timestamp = db.Column(db.DateTime, index=True, default=fake_clock.now)


class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    timestamp = db.Column(db.Float, index=True)
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
        user = db.session.get(User, self.user_id)
        return f'Productivity(username={user.username}, date={self.date}, dayshift_id={self.dayshift_id}, arrival={self.arrival}, staying={self.staying})'
    
    def __str__(self):
        return self.__repr__()

    # >>> from datetime import timedelta, time, datetime
    # >>> p = Productivity(user_id=3, date=datetime(day=13, month=4, year=2023).date(), dayshift_id=1, arrival=time(hour=8, minute=30, second=11), staying=timedelta(seconds=55))
    # >>> p = Productivity(user_id=4, date=datetime(day=13, month=4, year=2023).date(), dayshift_id=1, arrival=time(hour=8, minute=30, second=9), staying=timedelta(seconds=60))
    # >>> p = Productivity(user_id=5, date=datetime(day=13, month=4, year=2023).date(), dayshift_id=1, arrival=time(hour=8, minute=30, second=6), staying=timedelta(seconds=57))
    # [Productivity(username=intern, date=2023-04-12, dayshift_id=1, arrival=08:30:19.900000, staying=0:01:02.200000), Productivity(username=engineer1, date=2023-04-12, dayshift_id=1, arrival=08:30:07.533333, staying=0:00:29.932326), Productivity(username=engineer2, date=2023-04-12, dayshift_id=1, arrival=08:30:30.366667, staying=0:00:48.067581)]


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