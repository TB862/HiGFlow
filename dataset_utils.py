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

import tsl 

import numpy as np
import pandas as pd
import torch

from torch_geometric.utils import to_undirected

def _build_pems_graph_from_tsl(
                                    ds_name: str,
                                    root: str = "/home/tbai869/data/tsl",
                                    threshold: float = 0.00,
                                    gamma: float = 100.0,
                                    undirected: bool = True,
                                    add_distance: bool = False,
                                    include_self: bool = True,
                                    normalize_axis: int = 1,
                                ):
    try:
        from tsl.datasets import PeMS03, PeMS04, PeMS07, PeMS08
    except Exception as e:
        raise ImportError("torch-spatiotemporal (tsl) not installed or import failed") from e

    name = ds_name.upper()
    cls_map = {"PEMS03": PeMS03, "PEMS04": PeMS04, "PEMS07": PeMS07, "PEMS08": PeMS08}
    if name not in cls_map:
        raise ValueError(f"TSL graph not supported for {ds_name}. Supported: {list(cls_map)}")

    ds = cls_map[name]()

    # TSL gives (edge_index, edge_weight) in PyG layout
    edge_index, edge_weight = ds.get_connectivity(
        method="distance",
        gamma=gamma,
        threshold=threshold,
        include_self=include_self,
        normalize_axis=normalize_axis,
        layout="edge_index",
    )

    edge_index = torch.from_numpy(edge_index)
    edge_weight = torch.from_numpy(edge_weight)

    edge_index, edge_weight = to_undirected(edge_index, edge_weight.squeeze())

    return edge_index, edge_weight 


def open_data(config: ml.ConfigDict, ds_config: ml.ConfigDict, device=0):
    ds_nm = ds_config.name 

    make_graph = True 
    pems_edge_index = None 
    pems_edge_attr = None 

    if 'PEMS' in ds_config.name or 'LA' in ds_config.name:
        data_path = os.path.join(config.data_root, ds_nm, ds_nm.lower() + '.npz')
        data = np.load(data_path, allow_pickle=True)
        data = data['data'][:, :, 0]
        if make_graph:
            tsl_root = os.path.join(config.data_root, "TSL")
            pems_edge_index, pems_edge_attr = _build_pems_graph_from_tsl(
                ds_name=ds_config.name,
                threshold=0.0,
                gamma=getattr(ds_config, "edge_gamma", 100.0),
                undirected=getattr(ds_config, "edge_undirected", True),
                add_distance=getattr(ds_config, "edge_add_distance", False),
                include_self=False,
                normalize_axis=1,
            )
    else:
        data_path = os.path.join(config.data_root, ds_nm, ds_nm.lower() + '.csv')
        data = pd.read_csv(data_path)
        
    if 'date' in data:
        top_row = data['date']
    else:
        top_row = np.arange(1, len(data)+1, 1)
        if 'PEM' not in ds_config.name and 'LA' not in ds_config.name:
            top_row = data.values[:, 0]
            data = data.values[:, 1:]

    datetime_index = pd.to_datetime(top_row, format="mixed")
    time_stamps_enc = time_features(datetime_index).transpose(1, 0)
    
    data = np.array(data, dtype='float')

    train_ratio, val_ratio, test_ratio = ds_config.data_splits
    train, temp_data = train_test_split(data, test_size=1-train_ratio, shuffle=False)
    val, test = train_test_split(temp_data, test_size=0.5, shuffle=False)

    data_dict = {
                    'data': data,
                    'train': train,
                    'val': val,
                    'test': test,
                    'edge_index': pems_edge_index,
                    'edge_attr': pems_edge_attr
                }

    return data_dict
