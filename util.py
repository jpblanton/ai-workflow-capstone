from pathlib import Path
import os
import json

import pandas as pd

DATA_DIR = Path(os.environ.get('DATA_DIR'))
TS_DIR = Path(os.environ.get('TS_DIR'))
MODEL_DIR = Path(os.environ.get('MODEL_DIR'))
CUR_DIR = Path(os.environ.get('CUR_DIR'))

class BadCountryError(Exception):
    pass


def ingest_data(dir_):
    if type(dir_) == 'str':
        src = Path(dir_)
    else:
        src = dir_
    if not (dir_ / 'invoices.csv').exists():
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
        all_data.to_csv(dir_ / 'invoices.csv', index=False)
        return all_data
    else:
        data = pd.read_csv(dir_ / 'invoices.csv')
        data['date'] = pd.to_datetime(data['date']) #usually doesn't save well
        return data


def convert_to_ts(df, country='all', ts_data='ts_data'):
    if country != 'all':
        if country != 'all' and  country not in df['country'].unique():
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
    fp_out = os.path.join(os.getcwd(), DATA_DIR, f'{country}_by_day.csv')
    by_day.to_csv(fp_out, index=False)
    return by_day, fp_out

