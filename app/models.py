from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login
import datetime
import json
from time import time


class User(UserMixin, db.Model):
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(128), nullable=False)
    workshifts = db.relationship('RegisteredWorkshift', backref='user', lazy='dynamic')
    messages_received = db.relationship('Message', foreign_keys='Message.recipient_id', backref='recipient', lazy='dynamic')
    last_message_read_time = db.Column(db.DateTime)
    notifications = db.relationship('Notification', backref='user',
                                    lazy='dynamic')
    productivities = db.relationship('Productivity', backref='user',
                                    lazy='dynamic')

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
    

    def late(self):
        c = 0
        now = datetime.datetime.now()
        for record in self.productivities.all():
            if record.arrival:
                latency = datetime.datetime.combine(datetime.date.today(), record.arrival) - datetime.datetime.combine(datetime.date.today(), record.dayshift.start_time)
                if latency <= datetime.timedelta(0):
                    latency = None
            elif now.time() < record.dayshift.start_time:
                latency = None
            else:
                latency = min(now, datetime.datetime.combine(datetime.date.today(), record.dayshift.end_time)) - datetime.datetime.combine(datetime.date.today(), record.dayshift.start_time)
            if latency:
                c += 1
        return c
    

    def staying(self):
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
    timestamp = db.Column(db.Float, index=True, default=time)
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



    
    
    