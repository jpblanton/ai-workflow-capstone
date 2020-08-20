import sys
import os
import pickle
from pathlib import Path
import sqlite3

import pandas as pd
import numpy as np
import fbprophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics

from util import convert_to_ts
from util import ingest_data

# should add these to env when dockerized
outlier_min = .25
outlier_max = .75
DATA_DIR = Path(os.environ.get('DATA_DIR'))
TS_DIR = Path(os.environ.get('TS_DIR'))
MODEL_DIR = Path(os.environ.get('MODEL_DIR'))
CUR_DIR = Path(os.environ.get('CUR_DIR'))

def _load_model(country):
    mod_path = CUR_DIR / MODEL_DIR
    with open(mod_path / f'{country}_model.pkl', 'rb') as f:
        model = pickle.load(f)
        return model


def _save_model(country, model):
    assert model is not None
    mod_path = CUR_DIR / MODEL_DIR
    with open(mod_path / f'{country}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# need to add:
    # logging
        # not too hard, can just log some times to CSV, check w/sol guidance
    # unit testing
        # kind of a bitch probably, have to look some of it up
    # recording results/drift/etc
        # shouldn't be too hard
        # will inc script to run unittests


# assume we only have the data at first
# make_ts_data on the country data asked for
# save in ts_data folder (which we'll check before)
# can then store the model in models/ directory
# takes a country string & np.datetime[ns]
def _train_model(country, date):

    """
    if any((CUR_DIR / MODEL_DIR).glob(f'{country}*')):
        # model is alreeady trained
        model = _load_model(country)
        return model, model.history.ds.min()
    """

    data = ingest_data(CUR_DIR / DATA_DIR)
    ts_data, fp = convert_to_ts(data, country=country)
    ts_data['ds'] = pd.to_datetime(ts_data.index)
    ts_data = ts_data.drop(['purchases', 'unique_invoices', 'streams', 'total_views'], axis=1)
    ts_data = ts_data.rename({'revenue': 'y'},axis=1)
    upp, low = ts_data['y'].quantile(outlier_min), ts_data['y'].quantile(outlier_max)
    ts_data.loc[~ts_data['y'].between(upp, low), 'y'] = np.NaN

    fin_data = ts_data[ts_data['ds'] < date]
    prophet = fbprophet.Prophet()
    prophet.fit(fin_data)
    _save_model(country, prophet)
    return prophet, ts_data['ds'].min()
