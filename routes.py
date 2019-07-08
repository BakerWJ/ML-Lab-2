from flask import Flask, render_template
from sklearn.externals import joblib
import pandas as pd
import random

app = Flask(__name__)

model = joblib.load('forest.pkl')

@app.route("/")
def index():
    data = pd.read_csv("Boston_March2018.csv")
    proptypedummy = pd.get_dummies(data["PROPTYPE"])
    data = pd.concat([data, proptypedummy], axis = 1)
    citydummy = pd.get_dummies(data["CITY"])
    data = pd.concat([data, citydummy], axis = 1)
    zipdummy = pd.get_dummies(data["ZIP"])
    data = pd.concat([data, zipdummy], axis = 1)
    data = data.fillna(-999)
    features = data[list(zipdummy.columns) + list(citydummy.columns) + ["BEDS", "BATHS", "SQFT", "AGE", "LOTSIZE", "CC", "MF", "SF"]]
    x = random.randint(1, 5000)
    address = data["ADDRESS"].iloc[x]
    prediction = model.predict([features.iloc[x]]).round(1)
    prediction = str(prediction).lstrip('[').rstrip(']').rstrip('.')
    return render_template("index.html", prediction=prediction, address=address)

if __name__=="__main__":
    app.run(debug=True)
