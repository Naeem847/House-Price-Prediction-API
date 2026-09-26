''' FastAPi Application for house Price Prediction
this FastApi Application loads a pretrainerd machine learning model
and provides a endpoint to make predictions on house prices
To run this Application
1.install dependencies:pip install FastApi uvicorn scikit-learn pandas
2.run theb server:uvicorn main:app --host 0.0.0.0 -- port 8000
3.Access the api documentation at: http://localHost:8000/dcs
'''
from fastapi import FastApi, HTTPEXception
from fastapi.responses import HTMLResponce
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import List
# initialize fastapi app
app=FastApi(
    title="House Price Prediction App",
    description="API for predicting house prices using a trained Linear Regression Model",
    version="1,0,0"
)
# load the trained model and scaler
try:
    with open('linear_regression_model.pkl','rb') as file:
        model=pickle.load()
    with open('scaler.pkl','rb') as file:
        scaler=pickle.load(file)
    print("model and scaler loaded successfully!")
except FileNotFoundError as e:
    print(f"error loding model or scaler:{e}")
    print("please ensures you have run the trainig notebook and save model files")

# define the input data model
class HouseFeatures(BaseModel):

# Features from the California Housing Dataset:
# MedInc: Median income in block group
# HouseAge: Median house age in block group
# AveRooms: Average number of rooms per household
# AveBedrms: Average number of bedrooms per household
# Population: Block group population
# AveOccup: Average number of household members
# Latitude: Block group latitude
# Longitude: Block group longitude

   MedInc:float
   HouseAge:float
   AvgRooms:float
   Avgbedrms:float  
   Population:float
   AveOccup:float
   Latitude:float
   longitude:float  

# Define the output model
class PredictionResponce(BaseModel):
    """Responce model for predictions"""
    predicted_price:float
    input_features:dict

@app.get("/",responce_class=HTMLResponce)  
async def root():
    """Roof endpoint with a user friendly HTML page"""
    return"""
    <
"""    