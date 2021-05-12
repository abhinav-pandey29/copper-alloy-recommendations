from flask import Flask, render_template, redirect
from flask_sqlalchemy import SQLAlchemy
import json

from forms import PredicitionRequestForm
from composition_predictor import run_inverse_model

DEBUG = True
PORT = 5000

app = Flask('__main__')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SECRET_KEY'] = b'895t78fgreuyf9vf39vfy3423ov2'

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


if __name__ == '__main__':
    db.create_all()
    app.run(debug=DEBUG, port=PORT)
