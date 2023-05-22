from flask import render_template, redirect, url_for, flash, request, session
from flask_login import login_user, current_user, logout_user, login_required
from werkzeug.urls import url_parse
from datetime import datetime

from app import db
from app.auth import bp
from app.models import User
from app.auth.forms import LoginForm, CreateAccountForm


@bp.route('/login', methods=['GET', 'POST'])
def login():
    
    if current_user.is_authenticated: # type: ignore
        return redirect(url_for('main.index'))
    
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('auth.login'))
        
        login_user(user, remember=form.remember_me.data)
        flash(f'Hello {user}!')
    
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('main.index')
        return redirect(next_page)
    
    return render_template('auth/login.html', title='Sign In', form=form)


@bp.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('main.index'))


@bp.route('create_account', methods=['GET', 'POST'])
@login_required
def create_account():

    if not current_user.role == 'admin': # type: ignore
        return redirect(url_for('main.index'))
    
    form = CreateAccountForm()

    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            role=form.role.data
        ) # type: ignore
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash(f'Account {user.__repr__()} created')
        return redirect(url_for('auth.create_account'))

    return render_template('auth/create_account.html', title='Create account', form=form)

