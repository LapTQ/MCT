from flask import Blueprint

bp = Blueprint('main', __name__) # type: ignore

from app.main import routes