import os

from flask import Flask, render_template, redirect
from flask_sqlalchemy import SQLAlchemy

from forms import PredicitionRequestForm
from composition_predictor import run_inverse_model


app = Flask(__name__)
env_config = os.getenv("APP_SETTINGS", "config.DevelopmentConfig")
app.config.from_object(env_config)

db = SQLAlchemy(app)


@app.route('/', methods=['GET', 'POST'])
def index():

    form = PredicitionRequestForm()
    results = None
    if form.validate_on_submit():

        from models import PredictionRequest

        property_name = form.property_name.data
        value = form.value.data

        request = PredictionRequest(kind=property_name, value=value)
        db.session.add(request)
        db.session.commit()

        results = run_inverse_model(property_name, value, 10)
        results = results.to_dict(orient='index')

        return results

    return render_template('index.html', form=form, results=results)

db.create_all()