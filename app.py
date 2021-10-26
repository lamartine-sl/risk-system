from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics 
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_values, outdated_packages_list
import json
import os
from scoring import score_model


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 
    
dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path'])

with open(output_model_path +'/trainedmodel.pkl', 'rb') as file:
    prediction_model = pickle.load(file)

def readpandas(filename):
    df = pd.read_csv(filename)
    return df

#######################Prediction Endpoint
@app.route("/prediction")
def prediction():        
    #call the prediction function you created in Step 3
    dataset = request.args.get('dataset')
    predicted = model_predictions(dataset)
    return str(predicted) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring")
def scoring():        
    #check the score of the deployed model
    f1score = score_model(output_model_path,'trainedmodel.pkl', test_data_path, 'testdata.csv')
    return str(f1score)#add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats")
def summarystats():        
    #check means, medians, and modes for each column
    summary_statistics = dataframe_summary()
    return str(summary_statistics)#return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics")
def diagnostics():        
    #check timing and percent NA values
    exe_time = execution_time()
    percent_nas = missing_values()
    installed_packages = outdated_packages_list()   
    return {'execution time':str(exe_time), 'percent_NAs':str(percent_nas), 'intalled packages':str(installed_packages)}#add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
