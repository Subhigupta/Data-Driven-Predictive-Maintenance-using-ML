
from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)#giving an indication to Flask Web app to generate UI part


#Load the classifier file
pickle_in=open('decision_tree_binary_clf.pkl','rb')
decision_tree_binary_clf=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict') #by default get method
def predict():

    """Let's Classify for failure in a machine 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: AirTemperature
        in: query
        type: number
        required: true
      - name: ProcessTemperature
        in: query
        type: number
        required: true
      - name: RotationalSpeed
        in: query
        type: number
        required: true
      - name: Torque
        in: query
        type: number
        required: true
      - name: ToolWear
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    
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

    prediction=decision_tree_binary_clf.predict(test_df)
    return "The predicted values is" + str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_file():

    """Let's Classify for failure in a machine 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    prediction=decision_tree_binary_clf.predict(df_test)
    return "The predicted values for the csv are" + str(list(prediction))
if __name__=='__main__':
    app.run(debug=True)