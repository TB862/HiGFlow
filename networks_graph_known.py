import torch 
import torch.nn as nn 
from cluster_utils import embed_feature_vectors, lift_feature_vectors
from torch_geometric.utils.dropout import dropout_edge
from torch.fft import fft, ifft
import ml_collections as ml 
import numpy as np 
from cluster_utils import * 
from mutils import * 
from typing import * 
from contextlib import nullcontext
import torch.fft
import torch.nn.functional as F 
from torch_geometric.nn.models import GCN 
from networks import FourierComponet, Lifter, DimensionChanger, TemporalMap, GLU, SpatialMap


import torch


class Reducer(DimensionChanger):
    def __init__(self, ncls1: int, ncls2: int, tbuffer: int, map_layer_num: int, domain_embed: bool):
        super(Reducer, self).__init__(ncls2, ncls2, tbuffer, False, map_layer_num)
        self.ncls1 = ncls1 
        self.ncls2 = ncls2

        hidden_dim = 64 
        self.node_embed = nn.Sequential(
                                            nn.Linear(self.ncls2, hidden_dim),
                                            nn.LeakyReLU(),
                                            nn.Dropout(p=0.1),
                                            nn.Linear(hidden_dim, self.ncls2)
        )

        self.embed_domain = domain_embed

    def forward(self, x, edge_index, edge_weight):
        batch, tbuffer, num_nodes, hidden_dim = x.shape 

        if self.embed_domain:
            node_embed = embed_nodes(self.ncls2, self.node_embed, self.time_node_embed, hidden_dim, batch, tbuffer, x.device)
        else:
            node_embed = torch.zeros((batch, tbuffer*self.ncls2, hidden_dim), device=x.device)  
        
        node_embed = node_embed.unflatten(1, (tbuffer, self.ncls2))
        
        x, (input_x, x_clusters) = embed_feature_vectors_alternate(x, edge_index, edge_weight, node_embed, self.module, self.ncls2)
   
        return x, x_clusters
    

class HiGFlowGraph(nn.Module):
    def __init__(self, config: ml.ConfigDict, ds_config: ml.ConfigDict):
        super(HiGFlowGraph, self).__init__()

        model_config = config.baselines.HiGFlow

        self.hybrid = model_config.hybrid 
        self.res = model_config.res 
        self.no_ret = model_config.no_ret 

        self.n_layers = model_config.n_layers 
        map_layer_num = model_config.map_layer_num
        num_nodes = ds_config.var_dim
        buffer_size = ds_config.buffer_size 

        embed_nlayer = model_config.embed_nlayer 
        hidden_dim = model_config.rnn_hidden_dim 

        self.tbuffer_size = ds_config.buffer_size 
        self.var_dim = ds_config.var_dim 
        self.horizon = ds_config.horizon 
        self.embed_size = model_config.embed_size 

        self.lde = model_config.lifter_domain_embed
        self.lie = model_config.lifter_image_embed 
        self.ede = model_config.embed_domain_embed

        self.n_clusters = [int(np.ceil(fac*num_nodes)) for fac in model_config.n_cluster_facs]


        lifters, reducers, components, projectors = [], [], [], [] 
        for n, ncls in enumerate(self.n_clusters):
            if n < self.n_layers - 1:
                reducers.append(Reducer(
                                            ncls, 
                                            self.n_clusters[n+1], 
                                            self.tbuffer_size,
                                            map_layer_num,
                                            self.ede
                                        ))
                lifters.append(Lifter(
                                            self.n_clusters[self.n_layers-n-1], 
                                            self.n_clusters[self.n_layers-n-2],
                                            self.tbuffer_size,
                                            map_layer_num,
                                            self.lde,
                                            self.lie
                                        ))
                
            in_ch = self.embed_size if n == self.n_layers - 1 else 2*self.embed_size 
            out_ch = self.tbuffer_size if n == 0 else self.embed_size 

            dim = 1 if len(self.n_clusters) == 1 else (1 if n == len(self.n_clusters) - 1 else 2)
            if self.res or self.no_ret:
                dim = 1
            components.append(FourierComponet(dim, out_ch, edge_dim=model_config.hidden_dim, tbuffer=ds_config.buffer_size, horizon=ds_config.horizon)) 

        self.graph_reducers = nn.ModuleList(reducers) 
        self.fourier_components = nn.ModuleList(components) 
        self.graph_projectors = nn.ModuleList(projectors)
        self.lifters = nn.ModuleList(lifters)

        hidden_dim = model_config.hidden_dim 

        self.forecast = nn.Sequential(
                                        nn.Linear(buffer_size, hidden_dim), 
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_dim, self.horizon)
                                    )

        scale = 0.05 

        self.node_proj_mat = nn.Parameter(scale*torch.randn(1, self.embed_size))
        self.edge_proj_mat = nn.Parameter(scale*torch.randn(1, self.embed_size))
        self.embed_mat = nn.Parameter(scale*torch.randn(self.embed_size, 1))

        self.dropout = nn.Dropout(p=0.00)

        self.edge_index = None 
        self.edge_attr = None 

        self.all_hidden_outputs = None 
        self.last_pred = None 

    def forward(self, x): 
        x = torch.matmul(x.unsqueeze(-1), self.node_proj_mat).squeeze()
        #edge_attr = self.edge_attr.squeeze()
        edge_attr = self.edge_attr.unsqueeze(-1) * self.edge_proj_mat

        features, graphs, edge_weights = [x], [self.edge_index], [edge_attr]        
        all_cluster_lists = []     

        for n, reducer in enumerate(self.graph_reducers):
            higher_x = features[-1]
            ei, ew = graphs[-1], edge_weights[-1]

            abst_x, clusters = reducer(higher_x, ei, ew)
            abst_graph, eweights = coarsen_edge_index(ei, ew, clusters)

            all_cluster_lists.append(clusters)
            graphs.append(abst_graph)
            edge_weights.append(eweights)
            features.append(abst_x)

        memoy_buffer = None 
        self.all_hidden_outputs = [] 
        for m, (fcmpnt, abst_x, abst_graph, ew) in enumerate(zip(reverse(self.fourier_components), reverse(features),\
                                                                  reverse(graphs), reverse(edge_weights))):

            if m == 0:
                message = abst_x
            else:
                lifter, x_cls = self.lifters[m - 1], all_cluster_lists[self.n_layers - m - 1]
                intermed_state = lifter(features[self.n_layers - m], memoy_buffer, x_cls)

                if self.res or self.no_ret:
                    message = intermed_state
                else:
                    message = torch.concat((abst_x, intermed_state), dim=3)

            memoy_buffer = fcmpnt(message, abst_graph, ew)

            if self.hybrid or self.res:
                memoy_buffer += abst_x

            context_handler = torch.no_grad() if m < len(edge_weights) - 1 else nullcontext()
            with context_handler:
                temp_pred = torch.matmul(memoy_buffer, self.embed_mat).squeeze()    
                self.last_pred = temp_pred 
                temp_pred = self.forecast(temp_pred.permute(0, 2, 1).contiguous())  
                temp_pred = temp_pred.transpose(1, 2)           
                temp_pred = self.dropout(temp_pred)                    
                self.all_hidden_outputs.append(temp_pred)         
 
        return self.all_hidden_outputs[-1] 