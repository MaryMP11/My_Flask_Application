from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField

class FlowerForm(FlaskForm):
    SepalLengthCm = StringField('Sepal Length Cm')
    SepalWidthCm = StringField('Sepal Width Cm')
    PetalLengthCm = StringField('Petal Length Cm')
    PetalWidthCm = StringField('Petal Width Cm')

    submit = SubmitField("Predict")