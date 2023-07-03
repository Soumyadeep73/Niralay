from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)

df = pd.read_csv("x.csv")


@app.route("/")
def index():
    return render_template("temp.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    sqft = int(request.form['sqft'])
    BHK = int(request.form['BHK'])
    locations = request.form['locations']

    with open('niralay.pickle', 'rb') as file:
        model = pickle.load(file)

    def predict_price(location, Sqft, BHK):
        loc_index = np.where(df.columns == location)[0]

        X = np.zeros(len(df.columns))
        X[0] = Sqft
        X[1] = BHK
        if loc_index >= 0:
            X[loc_index] = 1

        return model.predict([X])[0]

    data = predict_price(locations, sqft, BHK)
    return render_template('temp.html', data=data)


if __name__ == '__main__':
    app.run(debug=True, port=6299)
