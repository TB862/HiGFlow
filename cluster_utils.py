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

def coarsen_edge_index(
    edge_index: torch.LongTensor,
    edge_weight: torch.Tensor,
    I: torch.LongTensor,
    m: int = None,
    reduce: str = "sum",
    remove_self_loops: bool = True,
    undirected: bool = False):
    
    device = edge_index.device
    I = I.to(device)

    if edge_weight is not None and edge_weight.device != device:
        edge_weight = edge_weight.to(device)

    # Map nodes -> clusters [n] -> [m_out] via normalized labels
    uniq_clusters, I_norm = torch.unique(I, return_inverse=True)
    m_out = uniq_clusters.numel() if m is None else m
    if m is not None and m != uniq_clusters.numel():
        # Prefer consistent labeling from I; m_out unused but kept for compatibility
        m_out = uniq_clusters.numel()

    # Coarse endpoints
    src_c = I_norm[edge_index[0]]
    dst_c = I_norm[edge_index[1]]

    # Canonicalize undirected edges
    if undirected:
        a = torch.minimum(src_c, dst_c)
        b = torch.maximum(src_c, dst_c)
        src_c, dst_c = a, b

    coarse_ei = torch.stack([src_c, dst_c], dim=0)

    # Optionally drop self-loops before aggregation
    if remove_self_loops:
        mask = coarse_ei[0] != coarse_ei[1]
        coarse_ei = coarse_ei[:, mask]
        if edge_weight is not None:
            edge_weight = edge_weight[mask]

    E = coarse_ei.size(1)

    # Prepare weights with shape [E, D]
    if edge_weight is None:
        w = torch.ones(E, 1, device=device, dtype=torch.float32)
    else:
        w = edge_weight
        if w.dim() == 1:
            w = w.unsqueeze(1)          # [E] -> [E,1]
        elif w.dim() != 2:
            raise ValueError("edge_weight must be 1D or 2D (E or E×D).")

    # Unique coarse edges and inverse map from each fine edge -> unique bucket
    uniq_edges, inv = torch.unique(coarse_ei, dim=1, return_inverse=True)  # uniq_edges: [2, U]
    U = uniq_edges.size(1)
    D = w.size(1)

    # Aggregate per unique edge per feature column
    w_acc = torch.zeros(U, D, device=device, dtype=w.dtype)
    w_acc.index_add_(0, inv, w)  # sum over duplicates

    if reduce == "sum":
        w_out = w_acc
    elif reduce == "mean":
        counts = torch.bincount(inv, minlength=U).unsqueeze(1).clamp_min(1)
        w_out = w_acc / counts
    else:
        raise ValueError("reduce must be 'sum' or 'mean'.")

    # Optional: squeeze back to 1D if D==1 for backward compatibility
    if w_out.size(1) == 1:
        w_out = w_out.squeeze(1)

    return uniq_edges, w_out


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

def indices_from_inverse(inv, edges):
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    inv, perm = inv.flip([0]), perm.flip([0])
    indices = inv.new_empty(edges.size(0)).scatter_(0, inv, perm)

    return indices

def graclus_clustering(edge_index, edge_weight, n_clusters, num_nodes=None):
    # Returns a tensor of shape (batch, num_nodes) with values in (0, n_clusters - 1)
    # mapping each node to a cluster

    #edge_index, _ = remove_self_loops(edge_index)

    sorted_edges = torch.sort(edge_index, dim=0)[0]
    edges_t = sorted_edges.t()  
    unique_edges, inv = torch.unique(edges_t, dim=0, return_inverse=True, return_counts=False)
    indices = indices_from_inverse(inv, unique_edges)
    edge_index = unique_edges.t()
    edge_weight = edge_weight[indices]

    if num_nodes is None:
        num_nodes = len(torch.unique(edge_index))

    # Remove and reapply self-loop weights 
    ew, dw, di = remove_diag(edge_weight, edge_index)
    _, max_indices = torch.topk(ew, num_nodes - n_clusters, dim=0)
    edge_weight[di] = dw

    # Extract maximal edges
    maximal_edges = edge_index[:, max_indices]
    src_mx, dst_mx = maximal_edges[0], maximal_edges[1]
    
    adj = to_scipy_sparse_matrix(maximal_edges)
    _, component = sp.csgraph.connected_components(adj, directed=False)
    component = torch.from_numpy(component).to(edge_index.device, torch.int32)

    # Initialize cluster assignments
    node_indices = torch.arange(num_nodes, device=edge_index.device, dtype=component.dtype)
    
    node_indices[src_mx] = component[src_mx]
    node_indices[dst_mx] = component[dst_mx]

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


# for when the graph structure is known 
def embed_feature_vectors_alternate(
                            x: Tensor, 
                            edge_index: Tensor, 
                            edge_weight: Tensor, 
                            node_embed: Tensor, 
                            network: torch.nn.Module, 
                            n_cluster: int, 
                        ) -> Tuple[Tensor, List]:
    

    batch, num_nodes, dim = x.shape[0], x.shape[2], x.shape[-1]

    edge_weight = edge_weight.sum(dim=1)
    edge_weight = (edge_weight - edge_weight.mean(dim=0)) / (edge_weight.std(dim=0) + 0.05)
    indices = graclus_clustering(edge_index, edge_weight, n_cluster, num_nodes) 

    value_range = torch.arange(n_cluster, device=x.device)  
    cls_node_msk = (value_range == indices.unsqueeze(-1)).int().float()
    cls_node_msk = cls_node_msk.unsqueeze(0).unsqueeze(-1).expand(batch, num_nodes, n_cluster, dim)

    cluster_features = torch.einsum('bncf,btnf->btcf', cls_node_msk, x)
    cluster_features += node_embed

    out = network.spt_module(cluster_features.transpose(3, 2)).transpose(3, 2)

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