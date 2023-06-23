from flask import render_template, redirect, url_for, current_app, Response, flash, session, request, jsonify
from flask_login import login_required, current_user

from app.extensions import db
from app.main import bp
from app.models import User, RegisteredWorkshift, DayShift, Camera, Message, Notification, Productivity
from app.main.forms import EmptyForm, RegisterWorkshiftForm

from datetime import datetime, timedelta, date

##### START HERE #####
from app.main.tasks import cams, sm_oq     # TODO thay ten bien
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from mct.utils.pipeline import MyQueue, VisualizePipeline
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
    
    current_user.last_message_read_time = datetime.utcnow()
    current_user.add_notification('unread_message_count', 0)    # type: ignore
    db.session.commit()

    messages = current_user.messages_received.order_by(Message.timestamp.desc()).all()  # type: ignore

    return render_template('messages.html', messages=messages)


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
    
    weekday_id_to_name = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday'
    }
    
    records = Productivity.query.filter_by(user_id=user.id).order_by(Productivity.date.desc(), Productivity.dayshift_id.asc())
    productivity_report = []
    now = datetime.now()
    for record in records:
        if record.arrival:
            latency = datetime.combine(date.today(), record.arrival) - datetime.combine(date.today(), record.dayshift.start_time)
            if latency <= timedelta(0):
                latency = None
        elif now.time() < record.dayshift.start_time:
            latency = None
        else:
            latency = min(now, datetime.combine(date.today(), record.dayshift.end_time)) - datetime.combine(date.today(), record.dayshift.start_time)

        if now < datetime.combine(record.date, record.dayshift.end_time) or record.staying is None:
            staying = None
            staying_percent = None
        else:
            staying = record.staying
            staying_percent = record.staying / (datetime.combine(date.today(), record.dayshift.end_time) - datetime.combine(date.today(), record.dayshift.start_time))

            
        productivity_report.append(
            {
                'day': weekday_id_to_name[record.date.weekday()],
                'date': record.date,
                'dayshift': record.dayshift.name,
                'arrival': record.arrival,
                'latency': latency,
                'staying': staying,
                'staying_percent': staying_percent
            }
        )

    return render_template('productivity.html', user=user, productivity_report=productivity_report)
    


    




@bp.route('/view_cameras')
@login_required
def view_cameras():

    if current_user.role != 'manager':  # type: ignore
        return redirect(url_for('main.index'))
    
    cameras = Camera.query.all()
    
    ##### START HERE #####
    for cid, cv in cams.items():

        if 'pl_vis' not in cv:
            iq_vis_video = MyQueue(current_app.config['PIPELINE'].get('QUEUE_MAXSIZE'), name=f'IQ-Vis_Video<{cid}>')
            iq_vis_annot = MyQueue(current_app.config['PIPELINE'].get('QUEUE_MAXSIZE'), name=f'IQ-Vis_Annot<{cid}>')
            cv['pl_camera'].add_output_queue(iq_vis_video, iq_vis_video.name)
            sm_oq[cid] = iq_vis_annot
            pl_vis = VisualizePipeline(current_app.config['PIPELINE'], iq_vis_annot, iq_vis_video, name=f'Vis<{cid}>')
            cv['pl_vis'] = pl_vis
            pl_vis.start()

        iq_display = MyQueue(current_app.config['PIPELINE'].get('QUEUE_MAXSIZE'), name=f'IQ-Display<{cid}><USER_ID={current_user.id}><SESSION_CSRF={session["csrf_token"]}>')   # type: ignore
        cv['pl_vis'].add_output_queue(iq_display, iq_display.name)
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
                # while 'iq_display' not in cams[cam_id]:
                #     pass
                iq_display = cams[cam_id]['iq_display']
                while True:

                    if iq_display.empty():
                        continue
                    
                    frame = iq_display.get()
                    frame_img = frame['frame_img']
                    frame_img = cv2.resize(frame_img, (480, 240))
                    
                    imgbyte = cv2.imencode('.jpg', frame_img)[1].tobytes()

                    if time.time() - self.last_access > 3:
                        if 'pl_vis' in cams[cam_id]:
                            pl_vis = cams[cam_id]['pl_vis']
                            pl_vis.remove_output_queue(iq_display.name)
                            if len(pl_vis.output_queues) == 0:
                                cams[cam_id]['pl_camera'].remove_output_queue(pl_vis.video_queue.name)
                                del pl_vis
                                del sm_oq[cam_id]

                ##### END HERE #####
                        break

                    self.frame = (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + imgbyte + b'\r\n')
    
    return Response(FakeCamera().start(current_app._get_current_object(), int(cam_id)), # type: ignore
                    mimetype='multipart/x-mixed-replace; boundary=frame')

