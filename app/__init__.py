from flask import Flask, current_app
from config import Config
from app.extensions import db, migrate, login, bootstrap, moment


def create_app(config_class=Config):
    
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)
    login.init_app(app)
    bootstrap.init_app(app)
    moment.init_app(app)


    # from app.main.tasks import startup as main_startup
    # with app.app_context():
    #     main_startup()


    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')
    

    from app.main import bp as main_bp
    app.register_blueprint(main_bp)


    return app


from app import models
