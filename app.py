from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('logistic_regression_model2.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')



standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    output=0
    prediction=0
    if request.method == 'POST':
        Pregnancies = float(request.form['Pregnancies'])
        Glucose=float(request.form['Glucose'])
        BloodPressure=float(request.form['BloodPressure'])
        BloodPressure2=np.log(BloodPressure)
        BMI=float(request.form['BMI'])
        DiabetesPedigreeFunction=float(request.form['DiabetesPedigreeFunction'])

        prediction=model.predict([[Pregnancies,Glucose,BloodPressure,BMI,DiabetesPedigreeFunction]])
        output=int(prediction)
        if prediction==0.0:
            return render_template('index.html',prediction_text="He is Not A Diabetes Patient.You can intake more sugar")
        else:
            return render_template('index.html',prediction_text="He is A Diabetes Patient, Please Take Care Of Your Sugar Level")
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(host = "0.0.0.0", port = 5000)
    #app.run(debug=True)

