from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login


class User(UserMixin, db.Model):
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(128), nullable=False)
    workshifts = db.relationship('RegisteredWorkshift', backref='user', lazy='dynamic')

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
    

@login.user_loader
def load_user(id):
    return User.query.get(int(id))


class DayShift(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(32))
    start_time = db.Column(db.Time)
    end_time = db.Column(db.Time)

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

    __table_args__ = (
        db.CheckConstraint(type.in_(['checkin', 'overlap', 'workarea'])),
    )

    def __repr__(self):
        primary_cam_num = db.session.get(Camera, self.primary_cam_id).num
        secondary_cam_num = db.session.get(Camera, self.secondary_cam_id).num
        return f'Region(type={self.type}, secondary_cam_num={primary_cam_num}, secondary_cam_num={secondary_cam_num}, points=...)'


class CameraMatchingPoint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    primary_cam_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)      # sourse
    secondary_cam_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)    # destination
    points = db.Column(db.String(10000), nullable=False)                                    # source, destination

    def __repr__(self):
        primary_cam_num = db.session.get(Camera, self.primary_cam_id).num
        secondary_cam_num = db.session.get(Camera, self.secondary_cam_id).num
        return f'CameraMatchingPoint(primary_cam_num={primary_cam_num}, secondary_cam_id={secondary_cam_num}, points=...)'


