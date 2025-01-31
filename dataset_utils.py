import ml_collections as ml 
import torch
from torch.utils.data import Dataset, DataLoader 
import os 
import numpy as np 
from typing import * 
from networks import * 
import pandas as pd 
from torch.nn.functional import normalize
from sklearn.model_selection import train_test_split
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from time_features import time_features


def min_max_normalize(tensor):
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    return (tensor - min_val) / (max_val - min_val)


class TimeSeries(Dataset):
    def __init__(self, df, window_size, horizon, interval=1):
        self.window_size = window_size
        self.interval = interval
        self.horizon = horizon

        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data = df
        self.data = min_max_normalize(self.data)

        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()

    def __getitem__(self, index):
        hi = self.x_end_idx[index]
        lo = hi - self.window_size
        train_data = self.data[lo: hi]
        target_data = self.data[hi:hi + self.horizon]
        
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)

        #x, y = normalize(x), normalize(y)

        return x, y

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx
    

def open_data(config: ml.ConfigDict, ds_config: ml.ConfigDict, device=0):
    ds_nm = ds_config.name 
    data_path = os.path.join(config.data_root, ds_nm, ds_nm.lower() + '.csv')

    data = pd.read_csv(data_path)  

    if 'date' in data:
        top_row = data['date'].values
    else:
        top_row =  np.arange(1, len(data)+1, 1)

    datetime_index = pd.to_datetime(top_row)
    time_stamps_enc = time_features(datetime_index).transpose(1, 0)

    data = data.select_dtypes(include=[np.number]).to_numpy()
    data = data.astype(np.float32) 
    data = np.where(np.isnan(data), 0.0, data)
    data = data[:, 0:min(300, data.shape[1])]

    concat_flag = False 

    if concat_flag:
        data = np.concatenate((data, time_stamps_enc), axis=-1)

    train_ratio, val_ratio, test_ratio = ds_config.data_splits
    train, temp_data = train_test_split(data, test_size=1-train_ratio, shuffle=False)
    val, test = train_test_split(temp_data, test_size=0.5, shuffle=False)   # test data might not work 

    data_dict = {
                    'data': data,
                    'train': train,
                    'val': val,
                    'test': test
    }

    return data_dict
