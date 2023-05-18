from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField
from app.models import DayShift


class EmptyForm(FlaskForm):
    submit = SubmitField('Submit')


class RegisterWorkshiftForm(FlaskForm):
    day = SelectField('Day', choices=[(i, i) for i in ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday')])
    shift = SelectField('Shift', choices=[('morning', 'morning'), ('afternoon', 'afternoon')])
    submit = SubmitField('Submit')
