import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os
import env

def check_file_exists(filename, query, url):
    ''' 
    This checks if a file does exist or not from directory, if not it read from mySQL
    '''
    if os.path.exists(filename):
        print('this file exists, reading from csv file')
    else:
        print('this file does not exist, read from sql, and export to csv file')
    df = pd.read_sql(query, url)
    df.to_csv(filename)
    
    return df

# caching titanic_db
def get_titanic_db():
    '''
    This function will check if create the create titanic_db information 
    '''
    url = env.get_db_url('titanic_db')
    query = 'SELECT * FROM passengers'
    
    filename = 'titanic_csv'
    
    df = check_file_exists(filename, query, url)
    
    return df

# caching iris_db
def get_iris_db():
    '''
    This function will check if create the create iris_db information 
    '''
    url = env.get_db_url('iris_db')
    
    query = '''SELECT *
            FROM species
                JOIN measurements
                    USING (species_id)'''
                    
    filename = 'iris_csv'
    
    df = check_file_exists(filename, query, url)
    
    return df

# caching telco_db
def get_telco_db():
    '''
    This function will check if create the create telco_db information 
    '''
    url = env.get_db_url('telco_churn')
    
    query = '''SELECT * 
            FROM customers
                JOIN contract_types
                    USING (contract_type_id)
                JOIN internet_service_types
                    USING (internet_service_type_id)
                JOIN payment_types
                    USING (payment_type_id)'''
                    
    filename = 'telco_csv'
    
    df = check_file_exists(filename, query, url)
    
    return df