import os
from mct.utils.pipeline import ConfigPipeline
basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'password-of-laptq'
    
    # TODO os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')    
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    PIPELINE = ConfigPipeline('data/recordings/2d_v4/YOLOv7pose_pretrained-640-ByteTrack-IDfixed/config_pred_mct_trackertracker_18.yaml')
    
    MAX_LATENCY = 100                   # seconds
    MAX_ABSENCE = 100                   # seconds
    GAP_SAVE_DB_TRACKING = 10           # seconds
    DETECTION_EXPIRE_TIME = 0.2         # seconds
    MAX_ABSENCE_FRAMES = 10             # frames