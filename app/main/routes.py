from flask import render_template, redirect, url_for, current_app, Response, flash, session, request, jsonify
from flask_login import login_required, current_user

from app.extensions import db
from app.main import bp
from app.models import User, RegisteredWorkshift, DayShift, Camera, Message, Notification, Productivity
from app.main.forms import EmptyForm, RegisterWorkshiftForm

from datetime import datetime, timedelta, date

##### START HERE #####
from app.extensions import monitor, fake_clock
import time
from threading import Thread
##### END HERE #####


@bp.route('/', methods=['GET', 'POST'])
@bp.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    return render_template('index.html', title='Home')


@bp.route('/user/<username>')
@login_required
def user(username):
    user = User.query.filter_by(username=username).first_or_404() # type: ignore
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


@bp.route('/_unregister_workshift/<day>/<dayshift_id>', methods=['POST'])
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
    

@bp.route('/view_weekly_schedule')
@login_required
def view_weekly_schedule():

    if current_user.role != 'manager':  # type: ignore
        return redirect(url_for('main.index'))
    
    workshifts = RegisteredWorkshift.query.all()
    week = {ds: {d: [] for d in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']}  for ds in ['morning', 'afternoon']}
    for ws in workshifts:
        week[ws.dayshift.name][ws.day].append(ws.user.username)
        
    return render_template('view_weekly_schedule.html', week=week)


@bp.route('/messages')
@login_required
def messages():

    if current_user.role != 'manager':  # type: ignore
        return redirect(url_for('main.index'))
    
    current_user.last_message_read_time = fake_clock.now()
    current_user.add_notification('unread_message_count', 0)    # type: ignore
    db.session.commit()

    messages = current_user.messages_received.order_by(Message.timestamp.desc()).all()  # type: ignore

    return render_template('messages.html', messages=messages, now=fake_clock.now())


@bp.route('/notifications')
@login_required
def notifications():
    since = request.args.get('since', 0.0, type=float)
    notifications = current_user.notifications.filter(  # type: ignore
        Notification.timestamp > since).order_by(Notification.timestamp.asc())
    return jsonify([{
        'name': n.name,
        'data': n.get_data(),
        'timestamp': n.timestamp
    } for n in notifications])


@bp.route('/productivity/<username>')
@login_required
def productivity(username):
    
    user = User.query.filter_by(username=username).first_or_404() # type: ignore
    if (user.username != current_user.username and current_user.role != 'manager') or user.role == 'admin':    # type: ignore
        return redirect(url_for('main.index'))
    
    records = Productivity.query.filter_by(user_id=user.id).order_by(Productivity.date.desc(), Productivity.dayshift_id.asc())
    productivity_report = [user.extract_productivity(r) for r in records]

    return render_template('productivity.html', user=user, productivity_report=productivity_report)
    

def get_display_key(cam_id, user_id, csrf_token):
    return f'Display-<cam_id={cam_id}, user_id={user_id}, session_csrf={csrf_token}>'


@bp.route('/view_cameras')
@login_required
def view_cameras():

    if current_user.role != 'manager':  # type: ignore
        return redirect(url_for('main.index'))
    
    cameras = Camera.query.all()

    for camera in cameras:
        key = get_display_key(camera.id, current_user.id, session['csrf_token'])    # type: ignore
        monitor.register_display(
            cam_id=camera.id,
            key=key
        )
        
    return render_template('view_cameras.html', cameras=cameras)


@bp.route('/_video_feed/<cam_id>')
@login_required
def video_feed(cam_id):
    
    cam_id = int(cam_id)

    # get display queue
    key = get_display_key(cam_id, current_user.id, session['csrf_token'])   # type: ignore
    while True:
        display_queue = monitor.get_display_queue(cam_id, key)
        if display_queue is not None:
            break
        time.sleep(1)


    class _FakeCamera:

        def __init__(self, app, cam_id, key, display_queue):
            self.frame = None
            self.last_access = 0

            self.app = app

            self.cam_id = cam_id
            self.key = key
            self.display_queue = display_queue


        def start(self):
            Thread(target=self._thread).start()
            while True:
                self.last_access = time.time()
                time.sleep(0.01)
                if self.frame is None:
                    continue
                yield self.frame

        
        def _thread(self):
            import cv2

            # stream video
            with self.app.app_context():
                while True:
                    # deligate the FPS responsibility to the Visualizer
                    item = self.display_queue.get(block=True)
                    img = item['frame_img']
                    img = cv2.resize(img, (480, 240))
                    
                    imgbyte = cv2.imencode('.jpg', img)[1].tobytes()

                    if time.time() - self.last_access > 2:
                        monitor.withdraw_display(self.cam_id, self.key)
                        break

                    self.frame = (b'--frame\r\n'
                                    b'Content-Type: image/jpeg\r\n\r\n' + imgbyte + b'\r\n')
                
    return Response(
        _FakeCamera(
            app=current_app._get_current_object(), # type: ignore
            cam_id=cam_id,
            key=key,
            display_queue=display_queue
        ).start(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

