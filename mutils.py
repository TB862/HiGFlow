import ml_collections as ml 
from torch.nn import MSELoss, L1Loss
import torch 
from typing import * 
from itertools import product
import os 
from torch_geometric.nn.models import GAT 
from torch_geometric.utils import get_laplacian
import torch.nn as nn 


def create_mapper(in_dim, out_dim, num_layer, hidden_dim=64):
    # creates lifting and embedding networks; also used in sensitivity analysis  
    if num_layer == 1:
        hidden_dim = out_dim 

    mapper = []
    in_feat_dim = in_dim 
    out_feat_dim = hidden_dim
    for lidx in range(num_layer):
        mapper.append(nn.Linear(in_feat_dim, out_feat_dim))
        if lidx < num_layer - 1:
            mapper.append(nn.LeakyReLU())
        else:
            mapper.append(nn.Dropout(p=0.1))
        in_feat_dim = out_feat_dim
        if lidx == num_layer - 2:
            out_feat_dim = out_dim 
    mapper = nn.Sequential(*mapper)

    return mapper  
    

def remove_diag(edge_weight, edge_index):
    mask = (edge_index[0] == edge_index[1])
    diag_weights = edge_weight[mask]
    edge_weight[mask] = -1

    return edge_weight, diag_weights, mask



def combine_config(c1, c2):
    for key, val in c2.items():
        c1[key] = val 
    
    return c1


def get_edge_index(mask: torch.tensor, num_nodes: int = 0):
    ei = torch.nonzero(mask).transpose(0, 1)
    if num_nodes > 0:
        ei = ei + num_nodes 

    return ei 


def stack_graphs(edge_index, edge_weights, var_dim, time_dim):
    num_nodes_offset = torch.arange(0, var_dim*time_dim, var_dim, device=edge_index.device)
    
    num_nodes_offset = num_nodes_offset.repeat_interleave(edge_index.shape[-1]).repeat(2, 1)
    edge_index = edge_index.repeat(1, time_dim) + num_nodes_offset
    edge_weights = edge_weights.repeat(time_dim)

    return edge_index, edge_weights 


def save_to_checkpoint(model: torch.nn.Module, save_dir: str):
    """
        Saves the model
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model, os.path.join(save_dir, 'model.pt'))


def cheb_polynomial(laplacian):
    N = laplacian.size(0)  # [N, N]
    laplacian = laplacian.unsqueeze(0)
    first_laplacian = torch.ones([1, N, N], device=laplacian.device, dtype=torch.float)
    second_laplacian = laplacian
    third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
    forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian

    multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
    
    return multi_order_laplacian


def mul_reshape(x, shape=None):
    if shape == None:
        shape = x.shape 
    return x.view(-1, shape[0]*shape[1])


def graph_fourier_transform(x, edge_index, edge_weights, inverse=False):
    num_nodes = x.shape[1] * x.shape[2]
    dense_lap = torch.zeros((num_nodes, num_nodes), device=x.device)
    lei, lew = get_laplacian(edge_index, edge_weights, normalization='sym')
    dense_lap[lei[0], lei[1]] = lew 

    chebp = cheb_polynomial(dense_lap)
    if not inverse:
        chebp = chebp.transpose(-1, -2)

    gft = torch.matmul(chebp, x.flatten(1, 2).transpose(0, 1))
    gft = gft.permute(2, 0, 1).contiguous()
    gft = gft.unflatten(-1, (x.shape[2], x.shape[1]))

    return gft, chebp


def to_numpy(x: torch.tensor):
    return x.cpu().detach().numpy()


def reverse(lst):
    for item in reversed(lst):
        yield item


def get_ds_config(config: ml.ConfigDict, ds_name: str) -> ml.ConfigDict:
    if ds_name == 'Weather':
        ds_config = config.weather
    elif ds_name == 'METR-LA':
        ds_config = config.metr_la
    elif ds_name == 'AirQuality':
        ds_config = config.air_qual
    elif ds_name == 'ECG':
        ds_config = config.ecg
    elif ds_name == 'Electricity':
        ds_config = config.electricity
    elif ds_name == 'Covid':
        ds_config = config.covid
    elif ds_name == 'Solar':
        ds_config = config.solar 
    elif ds_name == 'Traffic':
        ds_config = config.traffic 
    elif ds_name == 'ETTh1':
        ds_config = config.etth1 
    elif ds_name == 'ETTh2':
        ds_config = config.etth2 
    elif ds_name == 'ETTm1':
        ds_config = config.ettm1
    elif ds_name == 'ETTm2':
        ds_config = config.ettm2
    elif ds_name == 'ECL':
        ds_config = config.ecl
    else:
        raise ValueError('Invalid dataset name')

    return ds_config


def collect_metrics(model, loader, callables: Dict[AnyStr, Callable], device) -> Dict[AnyStr, float]:
    """
        Collects metrics from a series of callable metric functions
    """
    
    metrics = {kw: 0 for kw in callables}
    model.eval()

    for input, target in loader:
        input, target = input.to(device), target.to(device)
        preds = model(input)
        for it_nm, item_func in callables.items():
            stat = to_numpy(item_func(preds, target))
            metrics[it_nm] = stat 

    return metrics
    

def load_from_checkpoint(config: ml.ConfigDict, ds_name: str, nrepeats: int) -> Dict[AnyStr, Any]:   
    """
        Loads in the metrics of models from checkpoint 
    """

    model_names = config.baselines.names
    metric_names = config[ds_name.lower()].experiments.metrics
    metric_dict = {mn: {key: [] for key in metric_names} for mn in model_names}

    for mn, mi in product(model_names, range(nrepeats)):                                
        full_path = os.path.join('checkpoints', ds_name, mn+str(mi), 'metrics.txt')
        if not os.path.exists(full_path):
            print(f'Metric file path {full_path} not found')
            continue

        # read into metric dict 
        with open(full_path, 'r') as reader:
            for mline in reader.readlines():
                mline = mline.split(' ')
                key = str(mline[0])
                val = float(mline[1])
                metric_dict[mn][key].append(val) 

    return metric_dict 


def get_metric_functions(ds_config, device):
    metric_names = ds_config.experiments.metrics
    callables = {}
    for mn in metric_names:
        if mn == 'MSE':
            callables[mn] = MSELoss().to(device)
        elif mn == 'MAE':
            callables[mn] = L1Loss().to(device)
        else:
            raise ValueError()

    return callables

