from flask import Flask, render_template
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('etc.pkl')

@app.route("/")
def index():
    x = []
    for i in range(864):
        x.append(0)
    x[862] = 1  # sets property type to MF
    x[860] = 8350  # sets lot size
    x[859] = 118  # age of house in years
    x[858] = 4772  # square footage
    x[857] = 4.0  # bathrooms
    x[856] = 12  # bedrooms
    prediction = model.predict([x])
    prediction = str(prediction).lstrip('[').rstrip(']').rstrip('.')
    return render_template("index.html", prediction=prediction)

if __name__=="__main__":
    app.run(debug=True)
