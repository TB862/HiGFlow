import ml_collections as ml

###############
# FORECASTING #
###############

def weather_config():
    config = ml.ConfigDict() 
    config.name = 'Weather'
    config.save_path = 'Datasets/Weather'
    config.excluded_vars = []
    config.data_splits = [0.8, 0.1, 0.1]

    config.var_dim = 21
    config.buffer_size = 12 
    config.horizon = 3 

    return config


def metr_la_config():
    config = ml.ConfigDict() 
    config.name = 'METR-LA'
    config.save_path = 'Datasets/METR-LA'
    config.excluded_vars = []
    config.data_splits = [0.8, 0.1, 0.1]

    config.var_dim = 207
    config.buffer_size = 12 
    config.horizon = 3 

    return config

def covid_config():
    config = ml.ConfigDict() 
    config.name = 'Covid'
    config.save_path = 'Datasets/Covid'
    config.excluded_vars = []
    config.data_splits = [0.8, 0.1, 0.1]

    config.var_dim = 212 
    config.buffer_size = 12 
    config.horizon = 3 

    return config


def air_qual_config():
    config = ml.ConfigDict() 
    config.name = 'AirQuality'
    config.save_path = 'Datasets/AirQuality'
    config.excluded_vars = []
    config.data_splits = [0.8, 0.1, 0.1]

    config.var_dim = 15 
    config.buffer_size = 12 
    config.horizon = 3 

    return config


def ecg_config():
    config = ml.ConfigDict() 
    config.name = 'ECG'
    config.save_path = 'Datasets/ECG'
    config.excluded_vars = []
    config.data_splits = [0.8, 0.1, 0.1]

    config.var_dim = 140
    config.buffer_size = 12 
    config.horizon = 3 

    return config


def electricity_config():
    config = ml.ConfigDict() 
    config.name = 'Electricity'
    config.save_path = 'Datasets/Electricity'
    config.excluded_vars = []
    config.data_splits = [0.8, 0.1, 0.1]

    config.var_dim = 300
    config.buffer_size = 12 
    config.horizon = 3 

    return config

def traffic_config():
    config = ml.ConfigDict() 
    config.name = 'Traffic'
    config.save_path = 'Datasets/Traffic'
    config.excluded_vars = []
    config.data_splits = [0.8, 0.1, 0.1]

    config.var_dim = 300
    config.buffer_size = 12 
    config.horizon = 3 

    return config


def solar_config():
    config = ml.ConfigDict() 
    config.name = 'Solar'
    config.save_path = 'Datasets/Solar'
    config.excluded_vars = []
    config.data_splits = [0.8, 0.1, 0.1]

    config.var_dim = 137
    config.buffer_size = 12 
    config.horizon = 3 

    return config


def etth1_config():
    config = ml.ConfigDict() 
    config.name = 'ETTh1'
    config.save_path = 'Datasets/ETTh1'
    config.excluded_vars = []
    config.data_splits = [0.8, 0.1, 0.1]

    config.var_dim = 7
    config.buffer_size = 12 
    config.horizon = 3 

    return config

def etth2_config():
    config = ml.ConfigDict() 
    config.name = 'ETTh2'
    config.save_path = 'Datasets/ETTh2'
    config.excluded_vars = []
    config.data_splits = [0.8, 0.1, 0.1]

    config.var_dim = 7
    config.buffer_size = 12 
    config.horizon = 3 

    return config

def ettm1_config():
    config = ml.ConfigDict() 
    config.name = 'ETTm1'
    config.save_path = 'Datasets/ETTm1'
    config.excluded_vars = []
    config.data_splits = [0.8, 0.1, 0.1]

    config.var_dim = 7
    config.buffer_size = 12 
    config.horizon = 3 

    return config


def ettm2_config():
    config = ml.ConfigDict() 
    config.name = 'ETTm2'
    config.save_path = 'Datasets/ETTm2'
    config.excluded_vars = []
    config.data_splits = [0.8, 0.1, 0.1]

    config.var_dim = 7
    config.buffer_size = 12 
    config.horizon = 3 

    return config

def ecl_config():
    config = ml.ConfigDict() 
    config.name = 'ECL'
    config.save_path = 'Datasets/ECL'
    config.excluded_vars = []
    config.data_splits = [0.8, 0.1, 0.1]

    config.var_dim = 7
    config.buffer_size = 12 
    config.horizon = 3 

    return config
    

def _make_pems_config(name: str, var_dim: int = 1, buffer_size: int = 12, horizon: int = 3):
    cfg = ml.ConfigDict()
    cfg.name = name
    cfg.save_path = f"Datasets/{name}"   # change if your directory layout differs
    cfg.excluded_vars = []               # e.g., ['occupancy'] if you load multivariate but want to drop some
    cfg.data_splits = [0.8, 0.1, 0.1]         # train, val (test = remaining)
    cfg.var_dim = var_dim
    cfg.buffer_size = buffer_size
    cfg.horizon = horizon
    return cfg


def pems03_config():
    return _make_pems_config("PEMS03", var_dim=358, buffer_size=12, horizon=3)


def pems04_config():
    return _make_pems_config("PEMS04", var_dim=307, buffer_size=12, horizon=3)


def pems07_config():
    return _make_pems_config("PEMS07", var_dim=883, buffer_size=12, horizon=3)


def pems08_config():
    return _make_pems_config("PEMS08", var_dim=170, buffer_size=12, horizon=3)



#######################
# NODE CLASSIFICATION #
#######################

def cora_config():
    config = ConfigDict()
    config.name = 'Cora'
    config.in_channels = 1433
    config.num_classes = 7
    config.hidden_channels = 64
    config.num_nodes = 2708
    config.loss = 'cross_entropy'
    config.save_path = 'model_runs'
    config.metrics = ['Accuracy']

    return config


def pubmed_config():
    config = ConfigDict()
    config.name = 'PubMed'
    config.in_channels = 500
    config.num_classes = 3
    config.hidden_channels = 64
    config.num_nodes = 19717
    config.loss = 'cross_entropy'
    config.save_path = 'model_runs'
    config.metrics = ['Accuracy']

    return config


def citeseer_config():
    config = ConfigDict()
    config.name = 'Citeseer'
    config.in_channels = 3703
    config.num_classes = 6
    config.hidden_channels = 64
    config.num_nodes = 3327
    config.loss = 'cross_entropy'
    config.save_path = 'model_runs'
    config.metrics = ['Accuracy']

    return config


def computers_config():
    config = ConfigDict()
    config.name = 'Computers'
    config.in_channels = 767
    config.num_classes = 10
    config.hidden_channels = 128
    config.num_nodes = 13752
    config.loss = 'cross_entropy'
    config.save_path = 'model_runs'
    config.metrics = ['Accuracy']

    return config


def photo_config():
    config = ConfigDict()
    config.name = 'Photo'
    config.in_channels = 745
    config.num_classes = 8
    config.hidden_channels = 128
    config.num_nodes = 7650
    config.loss = 'cross_entropy'
    config.save_path = 'model_runs'
    config.metrics = ['Accuracy']

    return config


def wisconsin_config():
    config = ConfigDict()
    config.name = 'Wisconsin'
    config.in_channels = 1703
    config.num_classes = 5
    config.hidden_channels = 64
    config.num_nodes = 251
    config.loss = 'cross_entropy'
    config.save_path = 'model_runs'
    config.metrics = ['Accuracy']

    return config


def texas_config():
    config = ConfigDict()
    config.name = 'Texas'
    config.in_channels = 1703
    config.num_classes = 5
    config.hidden_channels = 64
    config.num_nodes = 183
    config.loss = 'cross_entropy'
    config.save_path = 'model_runs'
    config.metrics = ['Accuracy']

    return config
