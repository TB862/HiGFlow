import torch
import os
from config.model_config import *
from config.dataset_config import *


def graph_config():
    config = ml.ConfigDict()
    config.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    config.data_root = os.path.join('Datasets')    # default value, perhaps should be over-ridden

    config.weather = weather_config()
    config.metr_la = metr_la_config()
    config.ecg = ecg_config()
    config.electricity = electricity_config()
    config.air_qual = air_qual_config()
    config.solar = solar_config()
    config.traffic = traffic_config()
    config.covid = covid_config()
    config.etth1 = etth1_config()
    config.etth2 = etth2_config()
    config.ettm1 = ettm1_config()
    config.ettm2 = ettm2_config()
    config.ecl = ecl_config()

    config.experiments = ml.ConfigDict()                                                                                
    config.experiments.num_repeats = 1
    config.experiments.training_config = training_config() 
    config.experiments.training_config.device = config.device 

    config.baselines = ml.ConfigDict()
    
    config.baselines.model_names = ['FourierForecast']                 # baseline params might need to be model specific later

    config.baselines.FourierForecast = fourier_forecast_config()
    config.baselines.StemGNN = StemGNN_config()
    config.baselines.FourierGNN = FourierGNN_config()
    config.baselines.HGCN = HGCN_config()
    config.baselines.LSTM = LSTM_config() 
    config.baselines.AGCRN = AGCRN_config() 
    config.baselines.LSTNet = LSTNet_config() 
    config.baselines.iTransformer = config.baselines.iInformer = config.baselines.Reformer = \
        config.baselines.Flowformer = config.baselines.Flashformer= iTransformer_config() 
    config.baselines.FEDformer =  config.baselines.Autoformer =  config.baselines.Informer = FEDFormer_config()
    config.baselines.FreTS = ml.ConfigDict()
    config.baselines.DLinear = ml.ConfigDict()
    config.baselines.NLinear = ml.ConfigDict()
    config.baselines.GWNET = ml.ConfigDict()
    return config