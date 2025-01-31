from mutils import *
from itertools import product
import matplotlib.pyplot as plt
import yaml
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.logging import log
import ml_collections as ml 
import torch 
import torch.nn as nn 
from torch_geometric.data import InMemoryDataset
from time import time 
from typing import * 
import torch.nn.functional as F
from mutils import get_metric_functions, get_args, collect_metrics, save_to_checkpoint
from handler import train, test 
import os 
import numpy as np 
from functools import partial 

import sys
import os 


def band_limited_surrogate(signal, freq_lim):
    
    N = signal.size(0)
    freqs = torch.pi * torch.fft.fftfreq(N) / 2
    mask = (freqs >= -torch.pi/2) and (freqs <= freq_lim)
    invert_mask = ~mask

    freq_signal = torch.fft.fft(signal)
    sub_freq_signal = freq_signal[mask]
    sub_signal = torch.view_as_real(torch.fft.ifft(sub_freq_signal))

    indices = torch.range(len(mask)) + 1 
    sub_indices = indices[invert_mask]
    zero_entry = torch.tensor(0).view(1, 1, 1).expand(*indices.shape[0])
    diff = torch.concat((zero_entry, indices[..., 1:]), dim=2)
    repeats = indices - diff 

    rest_signal = torch.repeat_interleave(sub_signal, repeats, dim=-1)
    
    return rest_signal


class EnergyLoss(nn.Module):
    def __init__(self, freq_bands: List[Tuple[float, float]], use_attn):
        super(EnergyLoss, self).__init__()

        self.main_loss = nn.MSELoss(reduction='mean')
        self.freq_bands = freq_bands
        self.alphas = [nn.Parameter(torch.tensor(0.0)) for _ in range(len(freq_bands) - 1)]
        self.use_attn = use_attn 

    def compute_fourier_transform(self, signal: torch.tensor, freq_low: float = None, freq_high: float = None):
        batch, time, nodes = signal.shape

        # Compute FFT and normalize by sqrt(time)
        signal_fft = torch.fft.fft(signal, dim=1)
        signal_fft = torch.view_as_real(signal_fft).transpose(1, 2)
        signal_fft = torch.flatten(signal_fft, 2, 3).transpose(1, 2)
        signal_fft = signal_fft / torch.sqrt(torch.tensor(time))

        # Compute frequencies
        freqs = torch.fft.fftfreq(time, d=1 / time)
        freqs = 2 * torch.pi * freqs / time

        if freq_low is not None and freq_high is not None:
            # Restrict to specific frequency band
            mask = (freqs >= freq_low) & (freqs < freq_high)
            mask = mask.repeat(2)
            mask = mask.unsqueeze(0).unsqueeze(-1).expand(batch, -1, nodes)
            signal_fft[~mask] = 0.0

        return signal_fft, freqs

    def coarsen_tensor(self, tensor, nodes, target_nodes: int):
        factor = nodes / target_nodes

        if factor < 1.0:
            raise ValueError("Target nodes must be fewer than or equal to original nodes.")

        coarsened_tensor = F.interpolate(tensor, size=target_nodes, mode='linear', align_corners=False)
    
        return coarsened_tensor


    def forward(self, output: torch.Tensor, target: torch.Tensor, output_set: List[torch.Tensor]):
        main_loss = self.main_loss(output, target)

        freq_low = self.freq_bands[0]
        multi_resolution_loss = 0
        """for i, (freq_high, alpha, y) in enumerate(zip(self.freq_bands[1:], self.alphas, output_set[:-1])):
            # Compute Fourier transforms for target and output at resolution i
            #target_fft, _ = self.compute_fourier_transform(target, freq_low, freq_high)
            #y_fft, _ = self.compute_fourier_transform(y)

            # Compute MSE loss for the Fourier-transformed signals
            #target_fft = self.coarsen_tensor(target_fft, target_fft.shape[-1], y_fft.shape[-1])
            #mse_loss = self.main_loss(target_fft, y_fft)

            target_sub = self.coarsen_tensor(target, target.shape[-1], y.shape[-1])
            mse_loss = self.main_loss(target_sub, y)

            # Accumulate weighted loss            
            multi_resolution_loss += mse_loss"""

        if self.use_attn:
            return main_loss + multi_resolution_loss
        else:
            return main_loss 


def meta_train(
                    config: ml.ConfigDict,
                    ds_config: ml.ConfigDict,
                    models: Dict[AnyStr, List[nn.Module]],
                    dataset: Dict[AnyStr, Any], 
                    loss
            ) -> Dict:

    """
        Handles the repetition of experiment runs
    """

    num_repeats = config.experiments.num_repeats
    device = config.device

    meta_metrics = {mn: {'RMSE': [], 'MAPE': [], 'MAE': [], 'Dirchlet': []} for mn in models}

    for n, (mn, model) in product(range(num_repeats), models.items()):  
        model = model().to(device) 
        
        model_id = (mn+str(n))
        full_path = os.path.join(os.getcwd(), 'checkpoints', ds_config.name, model_id)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        train_data, val_data, test_data = dataset['train'], dataset['val'], dataset['test']
        training_config = config.experiments.training_config
        training_config.device = config.device 

        if mn == 'FourierForecast':
            bands = config.baselines[mn].freq_bands
            sample_rate = config.baselines[mn].sample_rate 
            loss = EnergyLoss(bands, sample_rate)
        else:
            loss = nn.MSELoss()

        train_stat, norm_stat = train(train_data, val_data, training_config, full_path, model, loss)
        torch.save(model, os.path.join(full_path, 'model'))

        mae, mape, rmse, dirch = test(model, test_data, training_config, full_path, norm_stat)
        meta_metrics[mn]['RMSE'].append(rmse)
        meta_metrics[mn]['MAE'].append(mae)
        meta_metrics[mn]['MAPE'].append(mape)
        meta_metrics[mn]['Dirchlet'].append(dirch)

        # save metrics
        np.savetxt(os.path.join(full_path, 'metrics.txt'), [[str(mem), str(val)] for mem, val in meta_metrics.items()], fmt='%s')


    return meta_metrics