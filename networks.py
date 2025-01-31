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


class SpatialMap(nn.Module):
    def __init__(self, spt_in: int, spt_out: int, tdim: int, dbl: bool, map_layer_num: int):
        super(SpatialMap, self).__init__()

        self.spt_in = spt_in 
        self.spt_out = spt_out 
        self.tdim = tdim 

        hidden_dim = 64 

        self.spt_module = create_mapper((dbl+1)*spt_in, spt_out, map_layer_num, hidden_dim=hidden_dim) 

        """self.spt_module = nn.Sequential(
                                        nn.Linear((dbl+1)*spt_in, hidden_dim),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.LeakyReLU(),
                                        nn.Dropout(p=0.1),
                                        nn.Linear(hidden_dim, spt_out)
        )"""

    def forward(self, x, y=None):
        if y is not None:
            x = torch.concat((x, y), dim=1)
            
        x = x.permute(0, 2, 1).contiguous()
        x = x.unflatten(2, (self.tdim, -1))
        x = self.spt_module(x)
        x = x.permute(0, 2, 3, 1).contiguous()

        return x 
    

class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))
    
    
class TemporalMap(nn.Module):
    # for use on the temporal dimension.
    def __init__(self, tmp_in: int, tmp_out: int, n_layer: int):
        super(TemporalMap, self).__init__()

        self.tbuffer = tmp_in 

        hidden_inc_fac = 5
        self.real_map, self.img_map = [], [] 
        for n in range(n_layer):
            if n == 0:
                in_ch = tmp_in 
                out_ch = tmp_in * hidden_inc_fac
            elif n == n_layer - 1:
                in_ch = tmp_in * hidden_inc_fac
                out_ch = tmp_out 
            else:
                in_ch = out_ch = tmp_in * hidden_inc_fac

            self.real_map.append(GLU(in_ch, out_ch))
            self.img_map.append(GLU(in_ch, out_ch))

            self.real_map.append(nn.Dropout(p=0.0))
            self.img_map.append(nn.Dropout(p=0.0))

        self.real_map = nn.Sequential(*self.real_map)
        self.img_map = nn.Sequential(*self.img_map)
        
        #self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        xfft = torch.view_as_real(torch.fft.fft(x.transpose(1, 3), dim=1)) # check dim 
        real, img = xfft[..., 0], xfft[..., 1]

        real = self.real_map(real)
        img = self.img_map(img)

        xfft = torch.concat((real.unsqueeze(-1), img.unsqueeze(-1)),  dim=-1)
        x = torch.fft.irfft(torch.view_as_complex(xfft), n=x.shape[1], dim=-1)
        #x = self.dropout(x)
        x = x.transpose(1, 3)

        return x 


class GraphProjector(nn.Module):
    def __init__(self, model_config: ml.ConfigDict, var_dim: int, buffer_size: int, embed_size):
        super(GraphProjector, self).__init__()

        self.attn_thrs = model_config.min_attn 
        self.num_heads = model_config.num_heads 
        self.var_dim = var_dim 
        self.buffer_size = buffer_size
        self.hyper_dim = var_dim

        scale = 0.00

        self.weight_key = nn.Parameter(scale*torch.randn(size=(embed_size, 1)))
        self.weight_query = nn.Parameter(scale*torch.randn(size=(embed_size, 1)))
        #nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        self.drop_edges = model_config.drop_edges 

    def graph_attention(self, input):
        bat, N, fea = input.shape    

        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)

        attention = key.transpose(1, 2) * query
        attention = torch.mean(attention, dim=0)
        attention = F.softmax(attention, dim=0)
        
        return attention

    def forward(self, x):        
        x = x.flatten(1, 2)
        edge_weights = self.graph_attention(x) 
        #print(torch.max(edge_weights, dim=0)[0])
        #exit()

        with torch.no_grad():
            # instead of masking by attention score, use the mask for the top K nodes, where 
            # K is 0.8 * num_nodes
            min_attn_score = torch.quantile(edge_weights.flatten(), self.attn_thrs)
            mask = (edge_weights > min_attn_score)
    
            nconn = torch.sum(mask, dim=0)
            where_alone = torch.nonzero((nconn == 0))
            best_conn = torch.argmax(edge_weights, dim=0)

            mask[best_conn[where_alone], where_alone] = True 
            edge_index = torch.nonzero(mask).transpose(0, 1)
            
            numeric_mask = mask.long()
            row_norm = numeric_mask*edge_weights
            row_norm = torch.sum(row_norm, dim=1)
            num_nonzero = torch.sum(numeric_mask, dim=1)
            row_norm = torch.repeat_interleave(row_norm, num_nonzero)

            edge_weights = edge_weights[mask] / row_norm 
        
        #edge_index, mask = dropout_edge(edge_index, p=self.drop_edges)
        #edge_weights = edge_weights[mask]
    
        return edge_index, edge_weights


class DimensionChanger(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, tdim: int, dbl: bool, map_layer_num: int):
        super(DimensionChanger, self).__init__()

        self.in_ch, self.out_ch = in_ch, out_ch 
        hidden_dim = 64     # max(model_config.hidden_dim, model_config.increase_factor * self.in_ch)

        self.module = SpatialMap(in_ch, out_ch, tdim, dbl, map_layer_num)
        
        self.time_node_embed = create_mapper(2*tdim, tdim, map_layer_num, hidden_dim=hidden_dim) 

        """self.time_node_embed = nn.Sequential(
                                                nn.Linear(2*tdim, hidden_dim),
                                                nn.LeakyReLU(),
                                                nn.Linear(hidden_dim, hidden_dim),
                                                nn.LeakyReLU(),
                                                nn.Dropout(p=0.1),
                                                nn.Linear(hidden_dim, tdim)
        )"""

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
        batch, tbuffer, _, hidden_dim = x.shape
        
        if self.embed_domain:
            node_embed = embed_nodes(self.ncls2, self.node_embed, self.time_node_embed, hidden_dim, batch, tbuffer, x.device)
        else:
            node_embed = torch.zeros((batch, tbuffer*self.ncls2, hidden_dim), device=x.device)  
        
        x, x_clusters = embed_feature_vectors(x, edge_index, edge_weight, node_embed, self.module, self.ncls2)
   
        return x, x_clusters
    

class Lifter(DimensionChanger):
    def __init__(self, ncls1: int, ncls2: int, tbuffer: int, map_layer_num: int, domain_embed: bool, image_embed: bool):
        super(Lifter, self).__init__(ncls1, ncls2, tbuffer, True, map_layer_num)

        self.ncls1 = ncls1 
        self.ncls2 = ncls2 

        hidden_dim = 64

        # Add dropout to feature_embedder
        self.feature_embedder = nn.Sequential(
            nn.Linear(ncls1, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),  # 50% dropout rate
            nn.Linear(hidden_dim, ncls2)
        )

        # Add dropout to node_embed
        self.node_embed = nn.Sequential(
            nn.Linear(self.ncls1, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),  # 50% dropout rate
            nn.Linear(hidden_dim, self.ncls1)  # should this be ncls1 -> ncls2 instead?
        )

        hidden_dim = max(2 * ncls2, 64)

        self.embed_mat = nn.Parameter(torch.zeros((64, 1)))
        self.domain_embed = domain_embed
        self.image_embed = image_embed
        
    def forward(self, x, hidden_state, y: List[torch.tensor]): 
        batch, tbuffer, _, hidden_dim = x.shape

        if self.domain_embed: 
            node_embed_abst = embed_nodes(
                                            self.ncls1, self.node_embed, self.time_node_embed, hidden_dim, batch, tbuffer, x.device
            )
        else:
            node_embed_abst = torch.zeros((batch, tbuffer*self.ncls1, hidden_dim), device=x.device)

        if self.image_embed:
            embed = self.feature_embedder(hidden_state.transpose(2, 3)).transpose(2, 3)
        else:
            embed = torch.zeros((batch, tbuffer, self.ncls2, hidden_dim), device=x.device)

        new_x = lift_feature_vectors(
                                        x, hidden_state, embed, node_embed_abst, y, self.module
        )
   
        return new_x


class FourierComponet(nn.Module):
    def __init__(self, ndim: int, out_ch: int): 
        super(FourierComponet, self).__init__()

        self.fourier_time_module = TemporalMap(12, 12, 3)

        hidden_size = 64
        self.gcn = GCN(ndim*hidden_size, ndim*hidden_size, 1, hidden_size) 

        self.out_ch = out_ch 

    def forward(self, x, edge_index, edge_weight):      
        batch, tbuffer, var_dim, hidden_dim = x.shape 
        edge_index, edge_weight = stack_graphs(edge_index, edge_weight, tbuffer*var_dim, batch)

        x = self.fourier_time_module(x)
        x = x.flatten(0, 2) 
        x = self.gcn(x, edge_index, edge_weight)
        x = x.unflatten(0, (batch, tbuffer, -1))

        return x


class FourierForecast(nn.Module):
    def __init__(self, config: ml.ConfigDict, ds_config: ml.ConfigDict):
        super(FourierForecast, self).__init__()

        model_config = config.baselines.FourierForecast

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
            projectors.append(GraphProjector(model_config.graph_projector, ncls, ds_config.buffer_size, model_config.embed_size))

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
            components.append(FourierComponet(dim, out_ch)) 

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

        self.proj_mat = nn.Parameter(scale*torch.randn(1, self.embed_size))
        self.embed_mat = nn.Parameter(scale*torch.randn(self.embed_size, 1))

        self.dropout = nn.Dropout(p=0.0)

        self.all_hidden_outputs = None 
        self.edge_indexs = None
        self.edge_weights = None  
        self.last_pred = None 

    def forward(self, x): 
        x = torch.matmul(x.unsqueeze(-1), self.proj_mat).squeeze()

        features, graphs, edge_weights = [x], [], []        
        all_cluster_lists = []      

        for n, projector in enumerate(self.graph_projectors):
            if n > 0:
                reducer, ei, ew = self.graph_reducers[n - 1], graphs[n - 1], edge_weights[n - 1]
                abst_x, clusters = reducer(higher_x, ei, ew)
                features.append(abst_x)
                all_cluster_lists.append(clusters)
                higher_x = abst_x
            else:
                higher_x = features[0]
            abst_graph, eweights = projector(higher_x)
   
            graphs.append(abst_graph)
            edge_weights.append(eweights)

        hidden_state = None 
        self.all_hidden_outputs = [] 
        for m, (fcmpnt, abst_x, abst_graph, ew) in enumerate(zip(reverse(self.fourier_components), reverse(features),\
                                                                  reverse(graphs), reverse(edge_weights))):
            if m == 0:
                message = abst_x
            else:
                lifter, x_cls = self.lifters[m - 1], all_cluster_lists[self.n_layers - m - 1]
                intermed_state = lifter(features[self.n_layers - m], hidden_state, x_cls)
                message = torch.concat((abst_x, intermed_state), dim=3)

            hidden_state = fcmpnt(message, abst_graph, ew)

            context_handler = torch.no_grad() if m < len(edge_weights) - 1 else nullcontext()
            with context_handler:
                temp_pred = torch.matmul(hidden_state, self.embed_mat).squeeze()    
                self.last_pred = temp_pred 
                temp_pred = self.forecast(temp_pred.permute(0, 2, 1).contiguous())  
                temp_pred = temp_pred.transpose(1, 2)           
                temp_pred = self.dropout(temp_pred)                    
                self.all_hidden_outputs.append(temp_pred)         

        self.edge_indexs = graphs 
        self.edge_weights = edge_weights
 
        return self.all_hidden_outputs[-1] 