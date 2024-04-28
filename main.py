from flask import Flask,request
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)

#Load the classifier file
pickle_in=open('random_forest_binary_clf.pkl','rb')
random_forest_binary_clf=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict') #by default get method
def predict():
    Air_temperature==request.args.get('Air temperature [K]')
    Process_temperature==request.args.get('Process temperature [K]')
    Rotational_speed==request.args.get('Rotational speed [rpm]')
    Torque==request.args.get('Torque [Nm]')
    Tool_wear==request.args.get('Tool wear [min]')

    prediction=random_forest_binary_clf.predict([[Air_temperature,
                                                  Process_temperature,Rotational_speed,Torque,
                                                  Tool_wear]])
    return "The predicted values is" + str(prediction)

if __name__=='__main__':
    app.run(debug=True)