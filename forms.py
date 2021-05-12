from flask_wtf import FlaskForm
from flask_wtf.recaptcha import validators
from wtforms import SelectField, FloatField, SubmitField
from wtforms.validators import DataRequired


class PredicitionRequestForm(FlaskForm):

    property_name = SelectField(
        'Choose Property',
        choices=['Thermal Conductivity', 'Tensile Strength'],
        validators=[DataRequired()]
    )
    value = FloatField(
        'Value of Selected Property',
        validators=[DataRequired()]
    )
    submit = SubmitField('Get Results')
