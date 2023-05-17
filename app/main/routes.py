from flask import render_template, redirect, url_for, request, current_app, Response
from flask_login import login_required, current_user

from app.main import bp
from app.models import User


@bp.route('/', methods=['GET', 'POST'])
@bp.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    return render_template('index.html', title='Home')


@bp.route('/user/<username>')
@login_required
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    if user.username != current_user.username and current_user.role != 'manager' or user.role == 'admin':    # type: ignore
        return redirect(url_for('main.index'))
    return render_template('user.html', user=user)


@bp.route('/view_staff_list')
@login_required
def view_staff_list():
    
    if current_user.role != 'manager': # type: ignore
        return redirect(url_for('main.index'))
    
    users = User.query.filter(User.role.in_(['intern', 'engineer'])).all()
    return render_template('view_staff_list.html', users=users)

