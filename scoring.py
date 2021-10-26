from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])

#################Function for model scoring
def score_model(output_model_path, model_name, test_dataset_path, test_dataset_name):
    """
    This function train model, load test data and calculate an F1 score. 
    The results is write in the latestscore.txt file
    """
    test_data_path = os.path.join(test_dataset_path, test_dataset_name) 
    testdata=pd.read_csv(test_data_path)
    X_test = testdata.drop(["corporation","exited"], axis=1)
    y_test = testdata["exited"].values.reshape(-1,1).ravel()
    
    with open(output_model_path +'/'+model_name, 'rb') as file:
        model = pickle.load(file)
    
    y_test_pred = model.predict(X_test)
    f1 = metrics.f1_score(y_test, y_test_pred)

    with open(output_model_path +'/latestscore.txt', 'w') as f:
        f.write(str(f1))
    
    return f1

if __name__ == "__main__":
    score_model(output_model_path,'trainedmodel.pkl', test_data_path, 'testdata.csv')