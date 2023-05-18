from flask import render_template, redirect, url_for, request, current_app, Response, flash
from flask_login import login_required, current_user

from app import db
from app.main import bp
from app.models import User, RegisteredWorkshift, DayShift
from app.main.forms import EmptyForm, RegisterWorkshiftForm


@bp.route('/', methods=['GET', 'POST'])
@bp.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    return render_template('index.html', title='Home')


@bp.route('/user/<username>')
@login_required
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    if (user.username != current_user.username and current_user.role != 'manager') or user.role == 'admin':    # type: ignore
        return redirect(url_for('main.index'))
    
    workshifts = RegisteredWorkshift.query.filter_by(user_id=user.id).all()
    unregister_form_class = EmptyForm

    return render_template('user.html', user=user, workshifts=workshifts, unregister_form_class=unregister_form_class)


@bp.route('/view_staff_list')
@login_required
def view_staff_list():
    
    if current_user.role != 'manager': # type: ignore
        return redirect(url_for('main.index'))
    
    users = User.query.filter(User.role.in_(['intern', 'engineer'])).all()
    return render_template('view_staff_list.html', users=users)


@bp.route('/register_workshift', methods=['GET', 'POST'])
@login_required
def register_workshift():

    if current_user.role not in ['intern', 'engineer']: # type: ignore
        return redirect(url_for('main.index'))
    
    form = RegisterWorkshiftForm()
    if form.validate_on_submit():

        workshift = RegisteredWorkshift.query.filter_by(
            user_id=current_user.id, # type: ignore
            day=form.day.data,
            dayshift_id=DayShift.query.filter_by(name=form.shift.data).first().id
        ).first()
        
        if workshift is not None:
            flash('Workshift already exists.')
            return redirect(url_for('main.register_workshift'))
        
        workshift = RegisteredWorkshift(
            user_id=current_user.id, # type: ignore
            day=form.day.data,
            dayshift_id=DayShift.query.filter_by(name=form.shift.data).first().id
        )
        db.session.add(workshift)
        db.session.commit()
        flash(f'Registered {workshift}')
        return redirect(url_for('main.user', username=current_user.username))   # type:ignore
    
    return render_template('register_workshift.html', form=form)


@bp.route('/unregister_workshift/<day>/<dayshift_id>', methods=['POST'])
@login_required
def unregister_workshift(day, dayshift_id):

    if current_user.role not in ['intern', 'engineer']: # type: ignore
        return redirect(url_for('main.index'))
    
    form = EmptyForm()
    if form.validate_on_submit():
    
        workshift = RegisteredWorkshift.query.filter_by(
            user_id=current_user.id,     # type: ignore
            day=day, 
            dayshift_id=dayshift_id
        ).first()
        
        if workshift:
            db.session.delete(workshift)
            db.session.commit()
            flash(f'Unregisterd day={day}, dayshift_id={dayshift_id}')
        
    return redirect(url_for('main.user', username=current_user.username)) # type: ignore
    


