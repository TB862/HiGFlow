import os 
import sys 

import argparse
import json
import torch
import numpy as np
from config.all_config import *
import torch.nn as nn 

from mutils import get_ds_config, load_from_checkpoint 
from train import meta_train 
from networks import HiGFlow
from networks_graph_known import HiGFlowGraph
from dataset_utils import open_data 


def create_models(config: ml.ConfigDict, ds_config: ml.ConfigDict):
    model_names = config.baselines.model_names 
    horizon = config.experiments.training_config.horizon
    buffer_size = config.experiments.training_config.window_size 
    models = {} 
    for mn in model_names:
        if mn == 'HiGFlow':
            models[mn] = lambda: HiGFlow(config, ds_config)

        elif mn == 'HiGFlowGraph':
            models[mn] = lambda: HiGFlowGraph(config, ds_config)

        else:
            raise ValueError('Invalid model name')

    return models 


def run_experiment(pargs):
    # Unpack arguments
    config_name = pargs.config
    ds_name = pargs.dataset
    do_train = pargs.train
    do_test = pargs.test
    model = pargs.model
    nrepeats = pargs.nrepeats 
    gpu = pargs.gpu
    nlayer = pargs.nlayer
    use_attn = pargs.use_attn 
    map_layer_num = pargs.map_layer_num 
    horizon = pargs.horizon 
    lr = pargs.lr 
    epochs = pargs.epoch
    cluster_fac = pargs.cluster_fac 
    lde = pargs.lde 
    lie = pargs.lie 
    ede = pargs.ede 

    # Collect configs, dataset
    if config_name == 'graph':
        config = graph_config()
    elif isinstance(config_name, ml.ConfigDict):
        config = config_name
    else:
        raise ValueError('Invalid configuration name')
    
    config.baselines.HiGFlow.lifter_domain_embed = lde 
    config.baselines.HiGFlow.lifter_image_embed = lie 
    config.baselines.HiGFlow.embed_domain_embed = ede 

    if nrepeats is not None:
        config.experiments.num_repeats = nrepeats
    
    if epochs is not None:
        config.experiments.training_config.epoch = epochs 

    if cluster_fac is not None:
        config.baselines.HiGFlow.n_cluster_facs = [1, cluster_fac]

    if pargs.res:
        config.baselines.HiGFlow.res = pargs.res  
    if pargs.hybrid:
        config.baselines.HiGFlow.hybrid = pargs.hybrid 
    if pargs.no_ret:
        config.baselines.HiGFlow.hybrid = pargs.no_ret 

    if nlayer is not None:
        config.baselines.HiGFlow.n_layers = nlayer 
        if nlayer == 1:
            config.baselines.HiGFlow.n_cluster_facs = [1]
        elif nlayer == 2:
            config.baselines.HiGFlow.n_cluster_facs = [1, 0.75]
        elif nlayer == 3:
            config.baselines.HiGFlow.n_cluster_facs = [1, 0.5, 0.25]
        elif nlayer == 4:
            config.baselines.HiGFlow.n_cluster_facs = [1, 0.75, 0.5, 0.375]
        elif nlayer == 5:
            config.baselines.HiGFlow.n_cluster_facs = [1, 0.75, 0.5, 0.375, 0.25]

    config.baselines.HiGFlow.use_attn = use_attn 
    config.baselines.HiGFlow.map_layer_num = map_layer_num 

    ds_config = get_ds_config(config, ds_name)

    if lr is not None:
        config.experiments.training_config.lr = lr 
    if horizon is not None:
        config.experiments.training_config.horizon = horizon 
        ds_config.horizon = horizon 

    if pargs.window_size is not None:
        config.experiments.training_config.window_size = pargs.window_size

    ds_config.horizon = config.experiments.training_config.horizon
    ds_config.buffer_size = config.experiments.training_config.window_size
    
    if model is not None:
        config.baselines.model_names = [model]
    
    if gpu is not None:
        if gpu in [0, 1, 4]:
            config.device = torch.device(f'cuda:{gpu}')
        elif gpu == -1:
            config.device = torch.device('cpu')

    data_dict = open_data(config, ds_config, config.device)

    loss = nn.MSELoss(reduction='mean') 

    # Run training and save state
    if do_train:
        models = create_models(config, ds_config)
        all_metrics = meta_train(config, ds_config, models, data_dict, loss)
    elif do_test:
        nrepeats = config.experiments.num_repeats
        models = load_from_checkpoint(config, ds_name, nrepeats)

    print(5*'\n')
    print(all_metrics)
    for model, metric_dict in all_metrics.items():
        mse = metric_dict['MAE']
        print(model, np.mean(mse), np.std(mse), end='\n')

    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        help='Dataset name (e.g., Traffic, Electricity, PEMS03).')

    parser.add_argument('--use_attn', type=bool, default=True,
                        help='Enable attention modules (True/False).')

    parser.add_argument('--map_layer_num', type=int, default=3,
                        help='Number of mapping layers to use.')

    parser.add_argument('--config', type=str, default='graph',
                        help='Configuration preset or config file key to load (e.g., graph).')

    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU index to use. Overrides the GPU setting in the config.')

    parser.add_argument('--nrepeats', type=int, default=None,
                        help='Number of repeated runs (e.g., for different seeds). Overrides config.')

    parser.add_argument('--checkpoint', type=str, default='checkpoints',
                        help='Checkpoint directory for saving/loading model states.')

    parser.add_argument('--train', action='store_true',
                        help='Train the model.')

    parser.add_argument('--lde', action='store_false', default=True,
                        help='Disable lifter domain embedding (enabled by default).')

    parser.add_argument('--lie', action='store_false', default=True,
                        help='Disable lifter image embedding (enabled by default).')

    parser.add_argument('--ede', action='store_false', default=True,
                        help='Disable embedding domain embedding (enabled by default).')

    parser.add_argument('--test', action='store_true',
                        help='Evaluate the model on the test split.')

    parser.add_argument('--model', type=str, default=None,
                        help='Model name to run. Overrides the model setting in the config.')

    parser.add_argument('--nlayer', type=int, default=None,
                        help='Number of model layers. Overrides the config setting.')

    parser.add_argument('--window_size', type=int, default=None,
                        help='Input sequence length (T_in). Overrides the config setting.')

    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Overrides the config setting.')

    parser.add_argument('--horizon', type=int, default=None,
                        help='Forecast horizon (T_out). Overrides the config setting.')

    parser.add_argument('--epoch', type=int, default=None,
                        help='Number of training epochs. Overrides the config setting.')

    parser.add_argument('--cluster_fac', type=float, default=None,
                        help='Clustering factor for hierarchical/coarsening modules. Overrides config.')

    parser.add_argument('--hybrid', action='store_true', default=False,
                        help='Enable hybrid architecture variant.')

    parser.add_argument('--res', action='store_true', default=False,
                        help='Enable residual-connection variant.')

    parser.add_argument('--no_ret', action='store_true', default=False,
                        help='Disable retention/memory mechanism (no-retention variant).')


    pargs = parser.parse_args()
    run_experiment(pargs)