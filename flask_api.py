
from flask import Flask, request, render_template, url_for, redirect, jsonify
import pandas as pd
import numpy as np
import pickle
from flasgger import Swagger
import io 
import sklearn

app = Flask(__name__)
Swagger(app) #giving an indication to Flask Web app to generate UI part

print(sklearn.__version__)
#Load the binary classifier files
pickle_in_bnry_rf = open('random_forest_binary_clf.pkl','rb')
random_forest_binary_clf = pickle.load(pickle_in_bnry_rf)

#load the multinominal classifier files
pickle_in_multi_xgboost = open('xgbost_multi_clf.pkl','rb')
xgboost_multi_clf = pickle.load(pickle_in_multi_xgboost)

pickle_in_bnry_xgboost = open('xgboost_binary_clf.pkl','rb')
xgboost_binary_clf = pickle.load(pickle_in_bnry_xgboost)

pickle_in_multi_rf = open('random_forest_multi_clf.pkl','rb')
random_forest_multi_clf = pickle.load(pickle_in_multi_rf)

# Load the label encoder
pickle_in_failure_type_encoder = open('failure_type_encoder.pkl','rb')
failure_type_encoder = pickle.load(pickle_in_failure_type_encoder)

def get_predict(AirTemperature,ProcessTemperature,RotationalSpeed,Torque,ToolWear):
    
    # AirTemperature = request.args.get('AirTemperature')
    # ProcessTemperature = request.args.get('ProcessTemperature')
    # RotationalSpeed = request.args.get('RotationalSpeed')
    # Torque = request.args.get('Torque')
    # ToolWear = request.args.get('ToolWear')

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
    test_df = pd.DataFrame(data,columns=columns,index=index_values)

    return test_df

@app.route('/input_predict_binary', methods=['POST'])
def input_predict_binary():
    
    AirTemperature = request.form["airTemperature"]
    ProcessTemperature = request.form["processTemperature"]
    ToolWear = request.form["toolWear"]
    RotationalSpeed = request.form["rotationalSpeed"]
    Torque = request.form["torque"]

    df_test = get_predict(AirTemperature, ProcessTemperature, RotationalSpeed, Torque, ToolWear)
    model = request.form.get('model') # Check which model button was clicked

    if model == "Random Forest":
      prediction = random_forest_binary_clf.predict(df_test)
      return "The predicted value for the user input is" + str(list(prediction))
    elif model == "Xgboost":
      prediction = xgboost_binary_clf.predict(df_test)
      return "The predicted value for the user input is" + str(list(prediction))
    
@app.route('/file_predict_binary', methods=['POST'])
def file_predict_binary():
    
    if request.method == 'POST':
        f = request.files.get("file")
        df_test = pd.read_csv(f)
        model = request.form.get('model') # Check which model button was clicked

        if model == "Random Forest":
          prediction = random_forest_binary_clf.predict(df_test)
          #print(prediction)
          return "The predicted values for the user inputs in csv file are" + str(list(prediction))
        
        elif model == "Xgboost":
          prediction = xgboost_binary_clf.predict(df_test)
          return "The predicted values for the user inputs in the csv file are" + str(list(prediction))

@app.route('/file_predict_multinomial', methods=['POST'])
def file_predict_multinomial():
    
    if request.method == 'POST':
        f = request.files.get("file")
        df_test = pd.read_csv(f)
        model = request.form.get('model')  # Check which model button was clicked

        if model == "Xgboost":
          prediction = xgboost_multi_clf.predict(df_test)
          predicted_classes = failure_type_encoder.inverse_transform(prediction) # Decode numerical pdictions
          return "The predicted values for the user inputs in the csv file are" + str(list(predicted_classes))
        
        elif model == "Random Forest":
          prediction = random_forest_multi_clf.predict(df_test)
          predicted_classes = failure_type_encoder.inverse_transform(prediction) # Decode numerical pdictions
          return "The predicted values for the user inputs in the csv file are" + str(list(predicted_classes))
        
@app.route('/input_predict_multinomial',methods=['POST'])
def input_predict_multinomial():
    
    AirTemperature = request.form["airTemperature"]
    ProcessTemperature = request.form["processTemperature"]
    ToolWear = request.form["toolWear"]
    RotationalSpeed = request.form["rotationalSpeed"]
    Torque = request.form["torque"]

    df_test = get_predict(AirTemperature,ProcessTemperature,RotationalSpeed,Torque,ToolWear)
    model = request.form.get('model')  # Check which model button was clicked

    if model == "Random Forest":
      prediction = random_forest_multi_clf.predict(df_test)
      predicted_classes = failure_type_encoder.inverse_transform(prediction)
      return "The predicted value for the user input is" + str(list(predicted_classes))
    
    elif model == "Xgboost":
      prediction = xgboost_multi_clf.predict(df_test)
      predicted_classes = failure_type_encoder.inverse_transform(prediction)
      return "The predicted value for the user input is" + str(list(predicted_classes))

@app.route('/')
@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/health', methods=['GET'])
def health_check():
    health_status = {
        "status": "healthy",
        "message": "The application is running smoothly."
    }
    return jsonify(health_status), 200

@app.route('/binary')
def binary():
    return render_template("binary.html")

@app.route('/multinomial')
def multinomial():
    return render_template("multinomial.html")

if __name__=='__main__':
    app.run(debug=True)