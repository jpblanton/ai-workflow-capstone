import os
import flask
from flask import Flask
from flask import request
import pandas as pd


os.environ['DATA_DIR'] = 'data'
os.environ['TS_DIR'] = 'ts_data'
os.environ['MODEL_DIR'] = 'models'
os.environ['CUR_DIR'] = os.getcwd()

app = Flask(__name__)
# have to make sure this imports right
from model import _train_model

@app.route('/')
def hello_world():
    return 'Hello, World!'


# a proper request will be formatted
# /predict/country=<all>&year=1999&month=12&day=12
@app.route('/predict/')
def model_predict():
    country = request.args.get('country')
    year = request.args.get('year')
    month = request.args.get('month')
    day = request.args.get('day')
    dt_str = year + '/' + month + '/' + day
    start_date = pd.to_datetime(dt_str)
    model, ind_start = _train_model(country, start_date)
    fut = model.make_future_dataframe(periods=30)
    forecast = model.predict(fut)
    rev = forecast.tail(30)['yhat'].sum()
    print(f'Estimated revenue for 30 days from {start_date} is {rev}')
    return flask.Response(str(rev))
