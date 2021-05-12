from app import db
from datetime import datetime


class PredictionRequest(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    kind = db.Column(db.String(50), nullable=False)
    value = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
