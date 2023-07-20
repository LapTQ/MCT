from flask import Flask
from app.extensions import db, migrate, login, bootstrap, moment, fake_clock, monitor
from config import Config


def create_app(config_class=Config):
    
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)
    login.init_app(app)
    bootstrap.init_app(app)
    moment.init_app(app)
    monitor.init_app(app, db, fake_clock)


    from app.main.tasks import startup as main_startup
    with app.app_context():
        main_startup()


    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')
    

    from app.main import bp as main_bp
    app.register_blueprint(main_bp)


    return app


from app import models
