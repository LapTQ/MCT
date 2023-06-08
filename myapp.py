from app import create_app, db
from app.models import User, DayShift, RegisteredWorkshift, Camera, Region, CameraMatchingPoint, Message, Notification, Productivity


app = create_app()


@app.shell_context_processor
def make_shell_context():
    return {
        'db': db,
        'User': User,
        'DayShift': DayShift,
        'RegisteredWorkshift': RegisteredWorkshift,
        'Camera': Camera,
        'Region': Region,
        'CameraMatchingPoint': CameraMatchingPoint,
        'Message': Message,
        'Notification': Notification,
        'Productivity': Productivity
    }

