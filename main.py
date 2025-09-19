import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from myFunctions import filter_data, create_feature


app = FastAPI()
model = joblib.load('data/models/car_rental_202509101613.pkl')


class Form(BaseModel):
    client_id: str
    event_category: str
    event_label: str
    event_action: str
    visit_date: str
    visit_time: str
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    client_id: str
    result: int


@app.get('/status')
def status():
    return "I'm OK!"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.model_dump()])
    y = model['model'].predict(filter_data(create_feature(df)))

    return {
        'client_id': df.client_id[0],
        'result': y[0]
    }



