import numpy as np 
import torch 
import torch.linalg as linalg
from typing import * 
from mutils import to_numpy 
from torch_geometric.nn.pool import graclus, knn, knn_graph, fps
from torch_geometric.utils import get_laplacian
from mutils import remove_diag
import torch.nn.functional as F


Tensor = torch.Tensor 


def embed_nodes(num_nodes, spt_network, t_network, hidden_dim, batch, tbuffer, device):
    node_indices = torch.arange(0, num_nodes, 1.0).to(device).view(1, 1, -1)
    node_embed = spt_network(node_indices).transpose(1, 2)
    node_embed = node_embed.unsqueeze(-1).repeat(batch, 1, hidden_dim, tbuffer)

    time_indices = torch.arange(0, tbuffer, 1.0).to(device).view(1, 1, 1, -1).repeat(batch, node_embed.shape[1], hidden_dim, 1)
    t_input = torch.concat((time_indices, node_embed), dim=3)
    node_embed = t_network(t_input)
    node_embed = node_embed.transpose(2, 3).flatten(1, 2)

    return node_embed


def swap_tensor_entries(tensor1, tensor2, mask):
    assert tensor1.shape == tensor2.shape == mask.shape
    tmp = torch.clone(tensor1)
    tensor1[mask] = tensor2[mask]
    tensor2[mask] = tmp[mask]


from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp


def graclus_clustering(edge_index, edge_weight, n_clusters, num_nodes=None):
    # Returns a tensor of shape (batch, num_nodes) with values in (0, n_clusters - 1)
    # mapping each node to a cluster

    if num_nodes is None:
        num_nodes = len(torch.unique(edge_index))

    # Remove and reapply self-loop weights
    edge_weights, dw, di = remove_diag(edge_weight, edge_index)
    _, max_indices = torch.topk(edge_weight, num_nodes - n_clusters)
    edge_weights[di] = dw

    # Extract maximal edges
    maximal_edges = edge_index[:, max_indices]
    src_mx, dst_mx = maximal_edges[0], maximal_edges[1]
    
    adj = to_scipy_sparse_matrix(maximal_edges)
    _, component = sp.csgraph.connected_components(adj, directed=False)
    component = torch.from_numpy(component).to(edge_index.device, torch.int32)

    # Initialize cluster assignments
    node_indices = torch.arange(num_nodes, device=edge_index.device, dtype=component.dtype)
    nsrc_uniq = len(torch.unique(src_mx))

    node_indices[src_mx] = component[:nsrc_uniq]
    node_indices[dst_mx] = component[nsrc_uniq:]

    return node_indices


def add_undirected_edge(edge_index: Tensor, i: int, j: int):
    forward_edge = torch.tensor((i, j), device=edge_index.device, dtype=edge_index.dtype) 
    backward_edge = torch.tensor((j, i), device=edge_index.device, dtype=edge_index.dtype) 

    edge_index = torch.concat((edge_index, forward_edge), dim=-1)
    edge_index = torch.concat((edge_index, backward_edge), dim=-1)

    return edge_index


def embed_feature_vectors(
                            x: Tensor, 
                            edge_index: Tensor, 
                            edge_weight: Tensor, 
                            node_embed: Tensor, 
                            network: torch.nn.Module, 
                            n_cluster: int, 
                        ) -> Tuple[Tensor, List]:
    
    tbuffer = x.shape[1]
    n_cluster *= tbuffer  # Scale clusters by the time buffer

    x = x.flatten(1, 2)  # Flatten time and features
    batch, num_nodes, dim = x.shape

    indices = graclus_clustering(edge_index, edge_weight, n_cluster, num_nodes) 

    value_range = torch.arange(n_cluster, device=x.device)  
    cls_node_msk = (value_range == indices.unsqueeze(-1)).int().float()
    cls_node_msk = cls_node_msk.unsqueeze(0).unsqueeze(-1).expand(batch, num_nodes, n_cluster, dim)

    cluster_features = torch.einsum('bijk,bik->bjk', cls_node_msk, x) 
    cluster_features += node_embed

    out = network(cluster_features)

    return out, (x, indices)


def lift_feature_vectors(
                            x: Tensor,                      
                            hidden_state: Tensor,   
                            feat_embed: Tensor,     
                            node_embed: Tensor, 
                            higher_cls: Tuple,  
                            network: torch.nn.Module
                        ) -> Tensor:

    x = x.flatten(1, 2)                             
    hidden_state = hidden_state.flatten(1, 2)       
    hidden_state += node_embed
    out = network(hidden_state, x)
    out += feat_embed

    return out 