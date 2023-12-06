import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from pydataset import data
import prepare

def prep_iris(iris_db):
    '''
    This function will be cleaning untransformed iris data into a clean iris data
    ** use acquire.get_iris_db() to get fresh dataframe**
    '''
    
    # dropping columns = species_id and measurement_id
    iris_db = iris_db.drop(columns=['species_id', 'measurement_id'])
    
    # renaming columns from species_name to species
    iris_db = iris_db.rename(columns={'species_name':'species'})
    
    return iris_db

def prep_titanic(df):
    """
    This function will be cleaning untransformed titanic data into a clean titanic data
    """
    #drop unncessary columns
    df = df.drop(columns = ['embarked', 'age','deck', 'class'])
    
    #made this a string so its categorical
    df.pclass = df.pclass.astype(object)
    
    #filled nas with the mode
    df.embark_town = df.embark_town.fillna('Southampton')
    
    return df

def prep_telco(telco_db):
    '''
    This function will be cleaning untransformed telco data into a clean a telco data
    can use acquire.get_telco_db() to get fresh dataframe
    '''
    
    # dropping columns
    telco_db = telco_db.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id', 'total_charges']) 
    
    # # checking for null value
    # telco_db.isnull().sum()
    
    # replacing all space character with 0.0
    telco_db = telco_db.replace(' ', '0.0')
    
    return telco_db

def splitting_data(df, col):
    '''
    This function will split the dataframe and send in target variable
    *** use dataframe and target variable TARGET VARIABLE: for iris is: species; titanic: survived; telco: churn
                
    '''

    #first split
    train, validate_test = train_test_split(df,
                     train_size = 0.6,
                     random_state = 123,
                     stratify = df[col]
                    )
    
    #second split
    validate, test = train_test_split(validate_test,
                                     train_size=0.5,
                                      random_state=123,
                                      stratify = validate_test[col]
                        
                                     )
    return train, validate, test