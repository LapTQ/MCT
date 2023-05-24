from flask import render_template, redirect, url_for, current_app, Response, flash, session
from flask_login import login_required, current_user

from app import db
from app.main import bp
from app.models import User, RegisteredWorkshift, DayShift, Camera
from app.main.forms import EmptyForm, RegisterWorkshiftForm

##### START HERE #####
from app.main.tasks import cams, config     # TODO thay ten bien
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from mct.utils.pipeline import MyQueue
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
    

@bp.route('/view_cameras')
@login_required
def view_cameras():

    if current_user.role != 'manager':  # type: ignore
        return redirect(url_for('main.index'))
    
    cameras = Camera.query.all()
    
    ##### START HERE #####
    for cid, cv in cams.items():
        name=f'IQ-Display<{cid}><USER_ID={current_user.id}><SESSION_CSRF={session["csrf_token"]}>' # type: ignore
        iq_display = MyQueue(config.get('QUEUE_MAXSIZE'), name=name)
        cv['pl_display'].add_output_queue(iq_display, name)
        cv['iq_display'] = iq_display
    ##### END HERE #####
    
    return render_template('view_cameras.html', cameras=cameras)


@bp.route('/_video_feed/<cam_id>')
def video_feed(cam_id):
    class FakeCamera:

        def __init__(self):
            self.frame = None
            self.last_access = 0


        def start(self, app, cam_id):
            Thread(target=self._thread, args=(app, cam_id)).start()
            while True:
                self.last_access = time.time()
                if self.frame is None:
                    continue
                yield self.frame

        
        def _thread(self, app, cam_id):
            import cv2

            with app.app_context():
                ##### START HERE #####
                msg = ''
                iq_display = cams[cam_id]['iq_display']
                while True:

                    if iq_display.empty():
                        continue
                    
                    frame = iq_display.get()
                    frame_img = frame['frame_img']
                    frame_img = cv2.resize(frame_img, (480, 240))
                    
                    cv2.putText(frame_img, msg, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, color=(0, 255, 0), thickness=3)
                    imgbyte = cv2.imencode('.jpg', frame_img)[1].tobytes()

                    if time.time() - self.last_access > 3:
                        cams[cam_id]['pl_display'].remove_output_queue(iq_display.name)
                ##### END HERE #####
                        break

                    self.frame = (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + imgbyte + b'\r\n')
    
    return Response(FakeCamera().start(current_app._get_current_object(), int(cam_id)), # type: ignore
                    mimetype='multipart/x-mixed-replace; boundary=frame')

