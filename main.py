import os 
import sys 

sys.path.append(os.path.abspath('higher_order_attn'))
sys.path.append(os.path.abspath(os.path.join('higher_order_attn', 'grand_src')))
sys.path.append(os.path.join(os.getcwd(), 'StemGNN'))
sys.path.append(os.path.join(os.getcwd(), 'StemGNN', 'utils'))
sys.path.append(os.path.join(os.getcwd(), 'iTransformer'))
sys.path.append(os.path.join(os.getcwd(), 'iTransformer', 'layers'))
sys.path.append(os.path.join(os.getcwd(), 'iTransformer', 'utils'))

sys.path.append(os.path.join(os.getcwd(), 'FourierGNN'))
#sys.path.append(os.path.join(os.getcwd(), 'FourierGNN', 'model'))
#sys.path.append(os.path.join(os.getcwd(), 'HGCN'))
#sys.path.append(os.path.join(os.getcwd(), 'FEDformer'))
#sys.path.append(os.path.join(os.getcwd(), 'FEDformer', 'models'))
#sys.path.append(os.path.join(os.getcwd(), 'FEDformer', 'layers'))



import argparse
import json
import torch
import numpy as np
from config.all_config import *
import torch.nn as nn 

from mutils import get_ds_config, load_from_checkpoint 
from train import meta_train 
from networks import FourierForecast
from dataset_utils import open_data 

from StemGNN.models.base_model import Model 
from FourierGNN.model.FourierGNN import FGN 
#from HGCN.model import * 

from baselines.AGCRN import AGCRN
from baselines.DLinear import Model as DLinear 
from baselines.NLinear import Model as NLinear 
from baselines.FreTS import Model as FreTS 
#from baselines.GWNET import gwnet
from baselines.LSTNet import Model as LSTNet 

from iTransformer.model.iTransformer import Model as iTransformer 
from iTransformer.model.Flashformer import Model as Flashformer  
#from iTransformer.model.Flowformer import Model as Flowformer  
#from iTransformer.model.Reformer import Model as Reformer  

#from FEDformer.models.Autoformer import Model as Autoformer 
#from FEDformer.models.Informer import Model as Informer  
#from FEDformer.models.FEDformer import Model as FEDformer  

#Autoformer = Informer = FEDformer = None 

def create_models(config: ml.ConfigDict, ds_config: ml.ConfigDict):
    model_names = config.baselines.model_names 
    horizon = config.experiments.training_config.horizon
    buffer_size = config.experiments.training_config.window_size 
    models = {} 
    for mn in model_names:
        if mn not in ['LSTM', 'GRU']: # quick fix 
            mc = config.baselines[mn]
            mc.horizon = horizon
            mc.buffer_size = buffer_size 
            mc.num_nodes = ds_config.var_dim
        if mn == 'FourierForecast':
            models[mn] = lambda: FourierForecast(config, ds_config)
        elif mn == 'StemGNN':
            models[mn] = lambda: Model(ds_config.var_dim, 2, buffer_size, 2, horizon=horizon)
        elif mn == 'FourierGNN':
            models[mn] = lambda: FGN(pre_length=horizon, embed_size=mc.embed_size, feature_size=mc.feature_size, seq_length=buffer_size, hidden_size=mc.hidden_size)
        elif mn == 'AGCRN':
            models[mn] = lambda: AGCRN(mc)
        elif mn == 'FreTS':
            models[mn] = lambda: FreTS(mc)
        elif mn == 'DLinear':
            models[mn] = lambda: DLinear(mc)
        elif mn == 'NLinear':
            models[mn] = lambda: NLinear(mc)
        elif mn == 'LSTNet':
            models[mn] = lambda: LSTNet(mc, horizon)
        #elif mn == 'GWNET':
        #    models[mn] = lambda: gwnet(config.device, ds_config.var_dim, in_dim=buffer_size, out_dim=horizon)
        elif mn in ['HGCN', 'LSTM', 'GRU', 'GWNET']:
            mc = config.baselines.HGCN
            if mn == 'HGCN':
                cls = H_GCN_wh
            elif mn == 'LSTM':
                cls = LSTM
            elif mn == 'GRU':
                cls = GRU
            elif mn == 'GWNET':
                cls = gwnet
            else:
                raise ValueError() 
            models[mn] = lambda: cls(
                                        config.device, 
                                        ds_config.var_dim, 
                                        mc.dropout,
                                        in_dim=ds_config.var_dim,
                                        length=buffer_size,
                                        out_dim=horizon,
                                        dilation_channels=ds_config.var_dim
                                    )
        elif mn == 'iTransformer':
            mc.pred_len = ds_config.horizon
            models[mn] = lambda: iTransformer(mc)
        elif mn == 'Flashformer':
            mc.pred_len = ds_config.horizon
            mc.enc_in = mc.dec_in = mc.c_out = ds_config.var_dim 
            models[mn] = lambda: Flashformer(mc)
        elif mn == 'Flowformer':
            mc.pred_len = ds_config.horizon
            mc.enc_in = mc.dec_in = mc.c_out = ds_config.var_dim 
            models[mn] = lambda: Flowformer(mc)
        elif mn == 'Reformer':
            mc.pred_len = ds_config.horizon
            mc.enc_in = mc.dec_in = mc.c_out = ds_config.var_dim 
            models[mn] = lambda: Reformer(mc)
        elif mn == 'FEDformer':
            mc.pred_len = ds_config.horizon
            mc.enc_in = mc.dec_in = mc.c_out = ds_config.var_dim 
            models[mn] = lambda: FEDformer(mc)
        elif mn == 'Autoformer':
            mc.pred_len = ds_config.horizon
            mc.enc_in = mc.dec_in = mc.c_out = ds_config.var_dim 
            models[mn] = lambda: Autoformer(mc)
        elif mn == 'Informer':
            mc.pred_len = ds_config.horizon
            mc.enc_in = mc.dec_in = mc.c_out = ds_config.var_dim 
            models[mn] = lambda: Informer(mc)
        else:
            raise ValueError('Invalid model name')

    return models 


def list_format(string):
    try:
        return json.loads(string)  
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError('Input must be a JSON list')


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
    
    config.baselines.FourierForecast.lifter_domain_embed = lde 
    config.baselines.FourierForecast.lifter_image_embed = lie 
    config.baselines.FourierForecast.embed_domain_embed = ede 

    if nrepeats is not None:
        config.experiments.num_repeats = nrepeats
    
    if epochs is not None:
        config.experiments.training_config.epoch = epochs 

    if cluster_fac is not None:
        config.baselines.FourierForecast.n_cluster_facs = [1, cluster_fac]

    if nlayer is not None:
        config.baselines.FourierForecast.n_layers = nlayer 
        if nlayer == 1:
            config.baselines.FourierForecast.n_cluster_facs = [1]
        elif nlayer == 2:
            config.baselines.FourierForecast.n_cluster_facs = [1, 0.75]
        elif nlayer == 3:
            config.baselines.FourierForecast.n_cluster_facs = [1, 0.75, 0.5]
        elif nlayer == 4:
            config.baselines.FourierForecast.n_cluster_facs = [1, 0.75, 0.5, 0.375]
        elif nlayer == 5:
            config.baselines.FourierForecast.n_cluster_facs = [1, 0.75, 0.5, 0.375, 0.25]

    config.baselines.FourierForecast.use_attn = use_attn 
    config.baselines.FourierForecast.map_layer_num = map_layer_num 


    ds_config = get_ds_config(config, ds_name)

    if lr is not None:
        config.experiments.training_config.lr = lr 
    if horizon is not None:
        config.experiments.training_config.horizon = horizon 
        ds_config.horizon = horizon 
    
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
    parser.add_argument('--dataset', type=str, help='The name of the dataset to run')
    parser.add_argument('--use_attn', type=bool, help='The type of configuration file', default=True)
    parser.add_argument('--map_layer_num', type=int, help='The type of configuration file', default=3)
    parser.add_argument('--config', type=str, help='The type of configuration file', default='graph')
    parser.add_argument('--gpu', type=int, default=None, help='Override model GPU in config file')
    parser.add_argument('--nrepeats', type=int, default=None, help='Override model GPU in config file')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint directory for loading models', default='checkpoints')
    parser.add_argument('--train', action='store_true', help='Option to train the model')
    parser.add_argument('--lde', action='store_false', default=True, help='lifter domain embed')
    parser.add_argument('--lie', action='store_false', default=True, help='lifter image embed')
    parser.add_argument('--ede', action='store_false', default=True, help='embed domain embed')
    parser.add_argument('--test', action='store_true', help='Option to test the model')
    parser.add_argument('--model', type=str, default=None, help='Override model in config file')
    parser.add_argument('--nlayer', type=int, default=None, help='Override model in config file')
    parser.add_argument('--lr', type=float, default=None, help='Override model in config file')
    parser.add_argument('--horizon', type=int, default=None, help='Override model in config file') 
    parser.add_argument('--epoch', type=int, default=None, help='Override model in config file')
    parser.add_argument('--cluster_fac', type=float, default=None, help='Override model in config file')


    pargs = parser.parse_args()
    run_experiment(pargs)