'''
utils.py
This file includes utility functions for retrieving hyperparameters,
generating file paths, and converting data formats.
'''

from importlib import import_module
import pandas as pd
import os
import ast



def get_config():
    '''
    Description:
        Retrieves hyperparameters from config file (uses configs/normal.py if none given)
    Input:
        None
    Output: 
        Hyperparameters form config file
    '''
    return import_module('config')


def make_path(type, chapter, name):
    '''
    Description: 
        Makes folder if it does not yet exist and constructs a path based on provided parameters
    Input:
        type (str): Type of file ("Data" or "Image")
        chapter (str): Name of chapter or section
        name (str): Name of the file
    Output:
        result_path (str): Constructed path
    '''
    os.makedirs(f"{type}/{chapter}", exist_ok=True) 
    params = get_config()
    if type == "Data":
        result_path = f"{type}/{chapter}/{name}_{params.n_steps}_{params.n_agents}_{params.n_rounds}.csv"
    else:
        result_path = f"{type}/{chapter}/{name}_{params.n_steps}_{params.n_agents}_{params.n_rounds}.png"

    return result_path


def string_to_list(data):
    '''
    Description: Converts string representations of lists in DataFrame columns to actual lists
    Input:
        data (DataFrame): Input DataFrame with columns containing string representations of lists
    Output:
        data (DataFrame): DataFrame with columns converted to lists
    '''
    if isinstance(data.values[0], str):
        data = data.apply(ast.literal_eval).tolist()
    return data


def xlsx_to_csv(input_folder, output_folder):
    '''
    Description: 
        Converts all XLSX files in the input folder to CSV files and saves them in the output folder
    Input:
        input_folder (str): Path to the folder containing XLSX files
        output_folder (str): Path to the folder where CSV files will be saved
    Output:
        None
    '''
    os.makedirs(output_folder, exist_ok=True)
    files = [file for file in os.listdir(input_folder) if file.endswith('.xlsx')]
    for file in files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file.replace('.xlsx', '.csv'))
        df = pd.read_excel(input_path)
        df.to_csv(output_path, index=False)
