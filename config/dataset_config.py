import ml_collections as ml


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