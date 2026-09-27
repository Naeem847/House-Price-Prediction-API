''' FastAPi Application for house Price Prediction
this FastApi Application loads a pretrainerd machine learning model
and provides a endpoint to make predictions on house prices
To run this Application
1.install dependencies:pip install FastApi uvicorn scikit-learn pandas
2.run theb server:uvicorn main:app --host 0.0.0.0 -- port 8000
3.Access the api documentation at: http://localHost:8000/dcs
'''
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import List
# initialize fastapi app
app=FastAPI(
    title="House Price Prediction App",
    description="API for predicting house prices using a trained Linear Regression Model",
    version="1,0,0"
)
# load the trained model and scaler
try:
    with open('linear_regression_model.pkl','rb') as file:
        model=pickle.load(file)
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

@app.get("/",response_class=HTMLResponse)  
async def root():
    """Roof endpoint with a user friendly HTML page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>House Price Prediction</title>

        <style>
            body {
                font-family: Arial;
                background-color: lightgray;
                margin:0;
                text-align: center;
                padding: 0;
            }
            .container{
            max-width:500px;
            margin:60px auto;
            background:#fff;
            border-radius:10px;
            box-shadow:0 2px 8px rgba(0,0,0,0.1);
            padding:32px 40px 32px 40px;
            text-align:center;
            }
            h1 {
                color: blue;
                margin-bottom:10px;
            }
            p{
            color:#444;
            margin-bottom:24px;
            }
            
            a.button {
                display:inline-block;
                padding: 12px 28px;
                background-color: green;
                color: white;
                text-decorator:none;
                border-radius: 5px;
                font-weight:bold;
                font-size:1.1em;
                transition: background 0.2s;
            }
            a.button:hover{
            background:#174ea6;
            }
        </style>
    </head>
    <body>
      <div class="container"
        <h1>House Price Prediction Api</h1>
        <p>
        Wellcome! this Api predict Californai house prices using a machine learning
         <b>Students:</b>Click bellow to explore and the test the Api endpoints!!
         </p>
        <a class="button" href="/docs"target="_blank">Open Api Documentation</a>
       </div>
    </body>
    </html>
    """
@app.post("/predict",response_model=PredictionResponce)
async def predict_house_price(features:HouseFeatures):
    """
    predict house price base on input feaetures.

    Args:

        features:House features object cntaining all required features.
    returns:
        Predictionresponce  with predicted price and input features

    """
    try:
        # convert input to numpy array:
        input_data=np.array([[
            features.MedInc,
            features.HouseAge,
            features.AvgRooms,
            features.Avgbedrms,
            features.Population,
            features.AveOccup,
            features.latitude,
            features.longitude
        ]])
# scale the input data using the saved scaler
        input_scaled=scaler.transform(input_data)
# make prediction
        prediction=model.predict(input_scaled)[0]
# convert prediction to a more readable format(multiply by )
        predicted_price=float(prediction*100000) #convert to price scale
        return PredictionResponce(
            predicted_price=predicted_price,
            input_features=features.dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500,details=f"Prediction error:{str(e)}")
# example usage and testing
if __name__=="__main__":
    import uvicorn
    # run the server
    uvicorn.run(app,host="127.0.0.1",port=8000) 