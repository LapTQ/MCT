from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from app.fakers import FakeClock


db = SQLAlchemy()
migrate = Migrate()
login = LoginManager()
login.login_view = 'auth.login' # type: ignore
bootstrap = Bootstrap()
moment = Moment()
fake_clock = FakeClock()

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from mct.utils.pipeline import Monitor

monitor = Monitor()

