import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    df_list = pd.DataFrame()
    input_path = os.getcwd()+'/'+input_folder_path+'/'
    output_path = os.getcwd()+'/'+output_folder_path+'/'
    filenames = os.listdir(input_path)

    for each_filename in filenames:
        df1 = pd.read_csv(input_path+each_filename)
        df_list = df_list.append(df1)
    
    df_final = df_list.drop_duplicates()
    df_final.to_csv(output_path+'finaldata.csv', index=False)
    
    with open(output_path+'ingestedfiles.txt', 'w') as f:
        f.write(str(filenames))

if __name__ == '__main__':
    merge_multiple_dataframe()
