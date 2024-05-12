
from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)#giving an indication to Flask Web app to generate UI part


#Load the binary classifier files
pickle_in_bnry_dt=open('decision_tree_binary_clf.pkl','rb')
decision_tree_binary_clf=pickle.load(pickle_in_bnry_dt)

pickle_in_bnry_xgboost=open('xgboost_binary_clf.pkl','rb')
xgboost_binary_clf=pickle.load(pickle_in_bnry_xgboost)

#load the multinominal classifier files
pickle_in_multi_lr=open('lr_multi_clf.pkl','rb')
lr_multi_clf=pickle.load(pickle_in_multi_lr)

pickle_in_multi_dt=open('dt_multi_clf.pkl','rb')
dt_multi_clf=pickle.load(pickle_in_multi_dt)

pickle_in_multi_rf=open('rf_multi_clf.pkl','rb')
rf_multi_clf=pickle.load(pickle_in_multi_rf)

def get_predict():
    
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

    return test_df

def post_predict():
    df_test=pd.read_csv(request.files.get("file"))

    return df_test

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict_bnry_dt') #by default get method
def predict_bnry_dt():

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

    test_df=get_predict()
    prediction=decision_tree_binary_clf.predict(test_df)
    return "The predicted values is" + str(prediction)

@app.route('/predict_file_bnry_dt',methods=["POST"])
def predict_file_bnry_dt():

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
    df_test=post_predict()
    prediction=decision_tree_binary_clf.predict(df_test)
    return "The predicted values for the csv are" + str(list(prediction))

@app.route('/predict_bnry_xgboost') #by default get method
def predict_bnry_xgboost():

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

    test_df=get_predict()
    prediction=xgboost_binary_clf.predict(test_df)
    return "The predicted values is" + str(prediction)

@app.route('/predict_file_bnry_xgboost',methods=["POST"])
def predict_file_bnry_xgboost():

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
    df_test=post_predict()
    prediction=xgboost_binary_clf.predict(df_test)
    return "The predicted values for the csv are" + str(list(prediction))

@app.route('/predict_multi_lr') #by default get method
def predict_multi_lr():

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

    test_df=get_predict()
    prediction=lr_multi_clf.predict(test_df)
    return "The predicted values is" + str(prediction)

@app.route('/predict_file_multi_lr',methods=["POST"])
def predict_file_multi_lr():

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
    df_test=post_predict()
    prediction=lr_multi_clf.predict(df_test)
    return "The predicted values for the csv are" + str(list(prediction))

@app.route('/predict_multi_dt') #by default get method
def predict_multi_dt():

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

    test_df=get_predict()
    prediction=dt_multi_clf.predict(test_df)
    return "The predicted values is" + str(prediction)

@app.route('/predict_file_multi_dt',methods=["POST"])
def predict_file_multi_dt():

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
    df_test=post_predict()
    prediction=dt_multi_clf.predict(df_test)
    return "The predicted values for the csv are" + str(list(prediction))

@app.route('/predict_multi_rf') #by default get method
def predict_multi_rf():

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

    test_df=get_predict()
    prediction=rf_multi_clf.predict(test_df)
    return "The predicted values is" + str(prediction)

@app.route('/predict_file_multi_rf',methods=["POST"])
def predict_file_multi_rf():

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
    df_test=post_predict()
    prediction=rf_multi_clf.predict(df_test)
    return "The predicted values for the csv are" + str(list(prediction))

if __name__=='__main__':
    app.run(debug=True)