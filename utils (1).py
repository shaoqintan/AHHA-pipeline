import numpy as np
import pandas as pd
import os
import logging
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import warnings
import psutil
from CFS import *
import sys


poly = PolynomialFeatures(2)

seed = 42
np.random.seed(seed)

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
pd.options.mode.chained_assignment = None  # default='warn'

poly = PolynomialFeatures(2)

seed = 42
np.random.seed(seed)



def create_dir(file):
    try:
        os.makedirs(file)
    except FileExistsError:
        pass
    
    
def get_memory_usage():
    memory_info = psutil.virtual_memory()
    return {
        "total": memory_info.total,
        "available": memory_info.available,
        "used": memory_info.used,
        "free": memory_info.free,
        "percent": memory_info.percent
    }

# Function to write memory usage information to a text file
def write_memory_usage_to_file(file_path):
    memory_info = get_memory_usage()
    with open(file_path, "a") as file:
        file.write("Memory Usage Information:\n")
        file.write(f"Total Memory: {memory_info['total']} bytes\n")
        file.write(f"Available Memory: {memory_info['available']} bytes\n")
        file.write(f"Used Memory: {memory_info['used']} bytes\n")
        file.write(f"Free Memory: {memory_info['free']} bytes\n")
        file.write(f"Memory Usage Percentage: {memory_info['percent']}%\n")



def setup_logger():
    # Get current date and time
    current_datetime = datetime.datetime.now()

    # Generate log filename with timestamp
    log_filename = f"log_{current_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    # Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    # Create a file handler and set the logging level
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Add the formatter to the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def get_processed_data(df_all_feature1, df_features, train_idx, val_idx, label):
        
    kinematic_columns = df_features
    
    scalar = MinMaxScaler()
    
    if (val_idx[0] != 0):
        df_train = df_all_feature1.loc[df_all_feature1['ID'].isin(train_idx)]
        df_train = df_train.reset_index().drop(columns=['index'])


        df_val = df_all_feature1.loc[df_all_feature1['ID'].isin(val_idx)]
        df_val = df_val.reset_index().drop(columns=['index'])
    else:
        
        df_train = df_all_feature1
        df_train = df_train.reset_index().drop(columns=['index'])

        df_val = df_all_feature1
        df_val = df_val.reset_index().drop(columns=['index'])
    
    df_train = df_train.dropna(axis=0, subset=kinematic_columns)
    df_val = df_val.dropna(axis=0, subset=kinematic_columns)
    
    if (label == 'ARAT'):
        all_columns = np.append(kinematic_columns, ['ARAT'])
    else:
        all_columns = np.append(kinematic_columns, ['ARAT', 'FM_score'])
    
    scalar.fit(df_train[all_columns])

    df_train[all_columns] = scalar.transform(df_train[all_columns])
    df_val[all_columns] = scalar.transform(df_val[all_columns])
        
    X_train = df_train[kinematic_columns]
    X_val = df_val[kinematic_columns]
    
    if (label == 'ARAT'):
        y_train = df_train['ARAT'].values
        y_val = df_val['ARAT'].values
        
    else:
        y_train = df_train['ARAT'].values + df_train['FM_score'].values
        y_val = df_val['ARAT'].values + df_val['FM_score'].values

   
    formula = ''
    for i in X_train.columns:
        formula += i
        formula += '+'

    formula = formula[:-1]
    formula = formula.replace(' ', '_')
    formula = formula.replace('^2', '2')

    X_val['ID'] = df_val['ID']
    X_train['ID'] = df_train['ID']
    
    return X_train, y_train, X_val, y_val, formula
