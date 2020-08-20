import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import sqlite3
from codetiming import Timer
conn = sqlite3.connect('/Users/jamesblanton/ai-workflow-capstone/invoices.db')


class BadCountryError(Exception):
    pass


def ingest_data(dir_):
    if type(dir_) == 'str':
        src = Path(dir_)
    else:
        src = dir_
    df_list = []
    for i, fp in enumerate(src.glob('*.json')):
        tmp = json.load(fp.open())
        df = pd.DataFrame.from_records(tmp)
        cols = set(df.columns)
        if 'total_price' in cols:
            df = df.rename(columns={'total_price': 'price'})
        if 'StreamID' in cols:
            df = df.rename(columns={'StreamID': 'stream_id'})
        if 'TimesViewed' in cols:
            df = df.rename(columns={'TimesViewed': 'times_viewed'})
        df_list.append(df)
    all_data = pd.concat(df_list)
    all_data['date'] = pd.to_datetime(
        all_data['year'].str.cat([all_data['month'],
                                  all_data['day']], sep='/'))
    all_data['invoice'].str.strip('AC')
    #all_data.to_sql('invoices', conn, if_exists='replace')
    all_data.to_csv('invoices.csv', index=False)
    return all_data


def convert_to_ts(df, country=None):
    if country is not None:
        if country not in df['country'].unique():
            raise BadCountryError('Country not found. Please try again')
        df = df[df['country'] == country]
    by_day = df.groupby('date').agg(
        purchases=pd.NamedAgg(column='invoice', aggfunc='count'),
        unique_invoices=pd.NamedAgg(column='invoice', aggfunc='nunique'),
        streams=pd.NamedAgg(column='stream_id', aggfunc='nunique'),
        total_views=pd.NamedAgg(column='times_viewed', aggfunc='sum'),
        revenue=pd.NamedAgg(column='price', aggfunc='sum'))
    tmp_ind = pd.date_range(start=by_day.index.min(), end=by_day.index.max(), name='date')
    missing_days = tmp_ind.difference(by_day.index)
    if len(missing_days) > 0:
        by_day = by_day.reindex(tmp_ind)
    #by_day.to_sql('daily_agg', conn, if_exists='replace')
    by_day.to_csv('by_day.csv')
    return by_day


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
