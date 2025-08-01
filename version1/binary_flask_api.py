
from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import sklearn

app=Flask(__name__)

#Load the classifier file
pickle_in=open('random_forest_binary_clf.pkl','rb')
random_forest_binary_clf=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict') #by default get method
def predict():

    # # Air_temperature=request.args.get('Air temperature')
    # # Process_temperature=request.args.get('Process temperature')
    # # Rotational_speed=request.args.get('Rotational speed')
    # # Torque=request.args.get('Torque')
    # # Tool_wear=request.args.get('Tool wear')

    AirTemperature = request.args.get('AirTemperature')
    ProcessTemperature = request.args.get('ProcessTemperature')
    RotationalSpeed = request.args.get('RotationalSpeed')
    Torque = request.args.get('Torque')
    ToolWear = request.args.get('ToolWear')

    data = {
    'AirTemperature': AirTemperature,
    'ProcessTemperature': ProcessTemperature,
    'RotationalSpeed': RotationalSpeed,
    'Torque': Torque,
    'ToolWear': ToolWear
    }
    index_values = [0]
    # Define column names
    columns = ['AirTemperature', 'ProcessTemperature', 'RotationalSpeed', 'Torque', 'ToolWear']
    test_df=pd.DataFrame(data,columns=columns,index=index_values)

    prediction=random_forest_binary_clf.predict(test_df)
    return "The predicted values is" + str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_file():
    df_test=pd.read_csv(request.files.get("file"))
    prediction=random_forest_binary_clf.predict(df_test)
    return "The predicted values for the csv are" + str(list(prediction))

if __name__=='__main__':
    app.run(debug=True)