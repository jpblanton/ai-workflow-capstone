import os
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import sqlite3
from codetiming import Timer
conn = sqlite3.connect('/Users/jamesblanton/ai-workflow-capstone/invoices.db')


# this shouldn't be needed since we're using fbprophet
def engineer_features(df, training=True):
    """
    for any given day the target becomes the sum of the next days revenue
    for that day we engineer several features that help predict the summed revenue
    
    the 'training' flag will trim data that should not be used for training
    when set to false all data will be returned
    """

    # extract dates
    dates = df.index.copy()

    # engineer some features
    eng_features = defaultdict(list)
    previous = [7, 14, 28, 70]  # [7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
    y = np.zeros(dates.size)
    for d, day in enumerate(dates):

        # use windows in time back from a specific date
        for num in previous:
            current = np.datetime64(day, 'D')
            prev = current - np.timedelta64(num, 'D')
            mask = np.in1d(dates, np.arange(prev, current, dtype='datetime64[D]'))
            eng_features["previous_{}".format(num)].append(df[mask]['revenue'].sum())

        # get get the target revenue    
        plus_30 = current + np.timedelta64(30, 'D')
        mask = np.in1d(dates, np.arange(current, plus_30, dtype='datetime64[D]'))
        y[d] = df[mask]['revenue'].sum()

        # attempt to capture monthly trend with previous years data (if present)
        start_date = current - np.timedelta64(365, 'D')
        stop_date = plus_30 - np.timedelta64(365, 'D')
        mask = np.in1d(dates, np.arange(start_date, stop_date, dtype='datetime64[D]'))
        eng_features['previous_year'].append(df[mask]['revenue'].sum())

        # add some non-revenue features
        minus_30 = current - np.timedelta64(30, 'D')
        mask = np.in1d(dates, np.arange(minus_30, current, dtype='datetime64[D]'))
        eng_features['recent_invoices'].append(df[mask]['unique_invoices'].mean())
        eng_features['recent_views'].append(df[mask]['total_views'].mean())

    X = pd.DataFrame(eng_features)
    # combine features in to df and remove rows with all zeros
    X.fillna(0, inplace=True)
    mask = X.sum(axis=1) > 0
    X = X[mask]
    y = y[mask]
    dates = dates[mask]
    X.reset_index(drop=True, inplace=True)

    if training:
        # remove the last 30 days (because the target is not reliable)
        mask = np.arange(X.shape[0]) < np.arange(X.shape[0])[-30]
        X = X[mask]
        y = y[mask]
        dates = dates[mask]
        X.reset_index(drop=True, inplace=True)

    #X.to_sql('features', conn, if_exists='replace')
    X.to_csv('features.csv', index=False)
    return(X, y, dates)


def _model_train():


if __name__ == '__main__':
    data_dir = Path('/Users/jamesblanton/ai-workflow-capstone/cs-train')
    t = Timer(name='class')
    t.start()
    data = ingest_data(dir_=data_dir)
    ts_data = convert_to_ts(data)
    t.stop()
    m, s = divmod(t.last, 60)
    h, m = divmod(m, 60)
    print("load time:", "%d:%02d:%02d" % (h, m, s))
    x, y, z = engineer_features(ts_data)

# Notes
# Roughly 190k, 23% of customer_id is NULL
# mostly in the UK
# all_data['invoice'].str.len().value_counts()
# shows that all invoices are either 6 or 7 characters
# sevens = all_data[all_data['invoice'].str.len() == 7]
# sevens['invoice'].str.slice(0,1)
# mostly all start w/'C' but 3 start with 'A'
# can remove letters according to github instructions
