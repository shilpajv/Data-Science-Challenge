
import sklearn
from random import shuffle
import numpy as np 
import pandas as pd 
import os
path= r'../data_interview_test.csv'
import xgboost
import pickle

def prepare_data(data):
    x = data[list(data.columns[4:])]
    x = xgboost.DMatrix(x)
    return x

def generate_output(preds,df):
    
    """ Generates output csv in decreasing order of probability of match"""
    res = df[df.columns[0:4]]
    res['scores'] = preds
    res.sort_values('scores', ascending= False, inplace = True)
    res.to_csv("Matched transactions output.csv")


if __name__ == "__main__":
   

    # Create xgb model
    model = xgboost.Booster()

    # Load trained model
    model_path = r"..artifacts/xgb_best.pkl"
    xgb_model_loaded = pickle.load(open(model_path, "rb"))

    # Read the test data 
    df = pd.read_csv(path, sep=":")
    x = prepare_data(df)

    # Generate Match probability scores
    preds=xgb_model_loaded.predict(x)

    # Saves the result in the decereasing order of likelihood of matching a receipt image
    generate_output(preds,df)
