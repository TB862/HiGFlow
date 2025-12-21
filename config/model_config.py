from ml_collections import ConfigDict
import numpy as np 


def training_config():
    config = ConfigDict()

    config.train = True
    config.evaluate = True
    config.window_size = 12
    config.horizon = 3
    config.train_length = 7.0
    config.valid_length = 2.0
    config.test_length = 1.0
    config.epoch = 50 
    config.lr = 1e-4
    config.multi_layer = 5
    config.validate_freq = 1
    config.batch_size = 32
    config.norm_method = 'z_score'
    config.optimizer = 'RMSProp'
    config.early_stop = False
    config.exponential_decay_step = 5
    config.decay_rate = 0.5
    config.dropout_rate = 0.5
    config.leakyrelu_rate = 0.2

    return config


def higflow_config():
    config = ConfigDict()
    config.n_layers = 2
    config.map_layer_num = 3
    config.n_cluster_facs = [1.0, 0.5]
    config.use_attn = False 
    config.stat_map = False 
    config.embed_nlayer = 1 
    config.hidden_dim = 64 
    config.embed_size = 64
    config.rnn_hidden_dim = 1
    config.graph_projector = graph_projector_config()
    config.freq_bands = np.linspace(-np.pi, np.pi, config.n_layers+1)
    config.sample_rate = 1000

    config.lifter_domain_embed = True 
    config.lifter_image_embed = False 
    config.embed_domain_embed = True 

    config.res = False 
    config.hybrid = False 
    config.no_ret = False 

    return config 


def graph_projector_config():
    config = ConfigDict()
    config.num_heads = 1 
    config.min_attn = 0.0
    config.drop_edges = 0.0
    
    return config 

def LSTNet_config():
    config = ConfigDict()

    config.data = None  # Required field, needs to be set
    config.model = "LSTNet"
    config.hidCNN = 100
    config.hidRNN = 100
    config.window = 24 * 7
    config.CNN_kernel = 6
    config.highway_window = 24
    config.clip = 10.0
    config.epochs = 100
    config.batch_size = 128
    config.dropout = 0.2
    config.seed = 54321
    config.gpu = None
    config.log_interval = 2000
    config.save = "model/model.pt"
    config.cuda = True
    config.optim = "adam"
    config.lr = 0.001
    config.horizon = 12
    config.skip = 24
    config.hidSkip = 5
    config.L1Loss = True
    config.normalize = 2
    config.output_fun = "sigmoid"

    return config

def StemGNN_config():
    config = ConfigDict()

    config.feature_size = 140
    config.seq_length = 12
    config.pre_length = 3
    config.embed_size = 128
    config.hidden_size = 256
    config.batch_size = 32
    config.exponential_decay_step = 5
    config.validate_freq = 1
    config.early_stop = False
    config.decay_rate = 0.5
    config.train_ratio = 0.7
    config.val_ratio = 0.2

    return config  

def FourierGNN_config():
    config = ConfigDict()

    config.feature_size = 140
    config.seq_length = 12
    config.pre_length = 3
    config.embed_size = 128
    config.hidden_size = 256
    config.batch_size = 32
    config.exponential_decay_step = 5
    config.validate_freq = 1
    config.early_stop = False
    config.decay_rate = 0.5
    config.train_ratio = 0.7
    config.val_ratio = 0.2

    return config


def LSTM_config():
    config = ConfigDict()

    config.device = 'cuda:0'
    config.data = 'data/XiAn_City'
    config.adjdata = 'data/XiAn_City/adj_mat.pkl'
    config.adjtype = 'doubletransition'
    config.seq_length = 12
    config.nhid = 32
    config.in_dim = 1
    config.num_nodes = 792
    config.batch_size = 64
    config.learning_rate = 0.001
    config.dropout = 0.3
    config.weight_decay = 0.0001
    config.epochs = 50
    config.print_every = 50
    config.force = False
    config.save = './garage/XiAn_City'
    config.expid = 1
    config.model = 'gwnet'
    config.decay = 0.92

    return config


def HGCN_config():
    config = ConfigDict()

    config.device = 'cuda:0'
    config.data = 'data/XiAn_City'
    config.adjdata = 'data/XiAn_City/adj_mat.pkl'
    config.adjtype = 'doubletransition'
    config.seq_length = 12
    config.nhid = 32
    config.in_dim = 1
    config.num_nodes = 792
    config.batch_size = 64
    config.learning_rate = 0.001
    config.dropout = 0.3
    config.weight_decay = 0.0001
    config.epochs = 50
    config.print_every = 50
    config.force = False
    config.save = './garage/XiAn_City'
    config.expid = 1
    config.model = 'gwnet'
    config.decay = 0.92

    return config

def DGraFormer_config(experiment_id: int = 0) -> ConfigDict:
    cfg = ConfigDict()

    # reproducibility
    cfg.random_seed = 202501

    # basic config
    cfg.is_training = True
    cfg.model_id = "test"
    cfg.model = "DGraFormer"

    # data loader
    cfg.data = "ETTh1"
    cfg.root_path = "data_provider/data/ETT"
    cfg.data_path = "ETTh1.csv"
    cfg.freq = "h"
    cfg.checkpoints = "./checkpoints/"

    # forecasting task
    cfg.seq_len = 96
    cfg.label_len = 48
    cfg.pred_len = 96

    # DCGL
    cfg.numpoint_win = 24
    cfg.w_bias = 0
    cfg.d_graph = 30
    cfg.d_gcn = 1
    cfg.w_ratio = 0.5
    cfg.mp_layers = 2

    cfg.predictor_dropout = 0.0
    cfg.patch_len = 8
    cfg.stride = 8
    cfg.revin = True
    cfg.affine = False
    cfg.subtract_last = 0  # 0: subtract mean; 1: subtract last

    # MTT
    cfg.n_vars = 7
    cfg.d_model = 16
    cfg.n_heads = 4
    cfg.e_layers = 1
    cfg.d_ff = 128
    cfg.dropout = 0.05
    cfg.embed = "timeF"
    cfg.activation = "gelu"
    cfg.do_predict = False

    # optimization
    cfg.num_workers = 4
    cfg.itr = 1
    cfg.train_epochs = 200
    cfg.batch_size = 128
    cfg.patience = 20
    cfg.learning_rate = 1e-4
    cfg.des = "test"
    cfg.lradj = "constant"
    cfg.pct_start = 0.3
    cfg.use_amp = False

    # GPU
    cfg.use_gpu = True
    cfg.gpu = 0
    cfg.use_multi_gpu = False
    cfg.devices = "0,1,2,3"
    cfg.device = f"cuda:{cfg.gpu}" if cfg.use_gpu else "cpu"
    cfg.test_flop = False

    return cfg


def AGCRN_config():
    config = ConfigDict()

    #num_nodes = 307
    #lag = 12
    #horizon = 12
    #val_ratio = 0.2
    #test_ratio = 0.2
    config.tod = False
    config.normalizer = 'std'
    config.column_wise = False
    config.default_graph = True

    config.input_dim = 1
    config.output_dim = 1
    config.embed_dim = 10
    config.rnn_units = 64
    config.num_layers = 2
    config.cheb_k = 2

    return config   


def iTransformer_config():
    config = ConfigDict()

    # Basic config
    config.is_training = 1
    config.model_id = 'test'
    config.model = 'iTransformer'  # Options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]   

    # Data loader
    config.data = 'custom'
    config.root_path = './data/electricity/'
    config.data_path = 'electricity.csv'
    config.features = 'M'  # Options: [M, S, MS]
    config.target = 'OT'
    config.freq = 'h'  # Options: [s, t, h, d, b, w, m]
    config.checkpoints = './checkpoints/'

    # Forecasting task
    config.seq_len = 12
    config.label_len = 12
    config.pred_len = 3

    # Model definition
    config.enc_in = 7
    config.dec_in = 7
    config.c_out = 7
    config.d_model = 512
    config.n_heads = 8
    config.e_layers = 2
    config.d_layers = 1
    config.d_ff = 2048
    config.moving_avg = 25
    config.factor = 1
    config.distil = True
    config.dropout = 0.1
    config.embed = 'timeF'  # Options: [timeF, fixed, learned]
    config.activation = 'gelu'
    config.output_attention = False
    config.do_predict = False

    # Optimization
    config.num_workers = 10
    config.itr = 1
    config.train_epochs = 10
    config.batch_size = 32
    config.patience = 3
    config.learning_rate = 0.0001
    config.des = 'test'
    config.loss = 'MSE'
    config.lradj = 'type1'
    config.use_amp = False

    # GPU
    config.use_gpu = True
    config.gpu = 0
    config.use_multi_gpu = False
    config.devices = '0,1,2,3'

    # iTransformer-specific
    config.exp_name = 'MTSF'  # Options: [MTSF, partial_train]
    config.channel_independence = False
    config.inverse = False
    config.class_strategy = 'projection'  # Options: [projection, average, cls_token]
    config.target_root_path = './data/electricity/'
    config.target_data_path = 'electricity.csv'
    config.efficient_training = False
    config.use_norm = True
    config.partial_start_index = 0

    return config

def FEDFormer_config():
    config = ConfigDict()
    
    # General settings
    config.is_training = 1
    config.task_id = 'test'
    config.model = 'FEDformer'
    
    # FEDformer specific settings
    config.version = 'Fourier'
    config.mode_select = 'random'
    config.modes = 64
    config.L = 3
    config.base = 'legendre'
    config.cross_activation = 'tanh'
    
    # Data loader settings
    config.data = 'ETTh1'
    config.root_path = './dataset/ETT/'
    config.data_path = 'ETTh1.csv'
    config.features = 'M'
    config.target = 'OT'
    config.freq = 'h'
    config.detail_freq = 'h'
    config.checkpoints = './checkpoints/'
    
    # Forecasting task settings
    config.seq_len = 12
    config.label_len = 12
    config.pred_len = 3
    
    # Model settings
    config.enc_in = 7
    config.dec_in = 7
    config.c_out = 7
    config.d_model = 512
    config.n_heads = 8
    config.e_layers = 2
    config.d_layers = 1
    config.d_ff = 2048
    config.moving_avg = [24]
    config.factor = 1
    config.distil = True
    config.dropout = 0.05
    config.embed = 'timeF'
    config.activation = 'gelu'
    config.output_attention = False
    config.do_predict = False
    
    # Optimization settings
    config.num_workers = 10
    config.itr = 3
    config.train_epochs = 30
    config.batch_size = 32
    config.patience = 3
    config.learning_rate = 0.0001
    config.des = 'test'
    config.loss = 'mse'
    config.lradj = 'type1'
    config.use_amp = False
    
    # GPU settings
    config.use_gpu = True
    config.gpu = 0
    config.use_multi_gpu = False
    config.devices = '0,1'
    
    return config


def AMD_config():
    config = ConfigDict()

    config.seq_len = 12
    config.pred_len = 3
    config.dropout = 0.0
    config.n_block = 4
    config.patch = 16
    config.mix_layer_num = 2
    config.mix_layer_scale = 1
    config.alpha = 0.5
    config.norm = True
    config.layernorm = True

    return config


def FilterTS_config():
    config = ConfigDict()

    config.task_name = 'forecasting'
    config.seq_len = 12
    config.pred_len = 3
    config.enc_in = 170          # input channels (c_in)
    config.c_out = 170           # output channels (c_out)
    config.mix_layer_num = 3
    config.mix_layer_scale = 2 
    config.d_model = 64        # model dimension
    config.alpha = 0.0 
    config.e_layers = 3        # number of frequency layers
    config.use_norm = True

    # frequency-domain and embedding parameters
    config.filter_type = 'all'   # or 'learnable', etc.
    config.quantile = 0.9            # quantile threshold for filtering
    config.bandwidth = 1            # filter bandwidth
    config.embedding = 'fourier_interpolate'  # options: 'fourier_interpolate', 'interpolate', 'complex'
    config.top_K_static_freqs = 16    # number of static frequency components to retain

    return config

from ml_collections import ConfigDict

def timemixerpp_config(dataset: str = "PEMS03",
                       seq_len: int = 12,
                       pred_len: int = 3,
                       down_sampling_layers: int = 1,
                       down_sampling_window: int = 2,
                       learning_rate: float = 0.003,
                       d_model: int = 128,
                       d_ff: int = 256,
                       batch_size: int = 32,     # matches your run.py
                       train_epochs: int = 10,
                       patience: int = 10) -> ConfigDict:
    """Config for TimeMixer++ on PEMS datasets, aligned with your run.py."""

    # Dataset-specific channel counts (enc_in/dec_in/c_out)
    enc_map = {
        "PEMS03": 358,
        "PEMS04": 307,
        # add others here when confirmed:
        # "PEMS07": xxx,
        # "PEMS08": xxx,
    }
    if dataset not in enc_map:
        raise ValueError(f"Unknown dataset '{dataset}'. Set enc_map entry first.")

    C = enc_map[dataset]


    cfg = ConfigDict()
    # --- Core task ---
    cfg.task_name = "long_term_forecast"
    cfg.data = "PEMS"
    cfg.features = "M"
    cfg.decomp_method = "moving_avg"
    cfg.moving_avg = 25 
    cfg.use_future_temporal_feature = 0

    # --- Sequence lengths ---
    cfg.seq_len = seq_len
    cfg.label_len = 0               # per your runs
    cfg.pred_len = pred_len

    # --- Architecture (matches CLI) ---
    cfg.e_layers = 5
    cfg.d_layers = 1
    cfg.factor = 3
    cfg.d_model = d_model
    cfg.d_ff = d_ff
    cfg.dropout = 0.0

    # Channel configuration
    cfg.enc_in = C
    cfg.dec_in = C
    cfg.c_out = C
    cfg.channel_independence = 0
    cfg.use_norm = 0

    # Downsampling pyramid
    cfg.down_sampling_layers = down_sampling_layers
    cfg.down_sampling_method = "avg"
    cfg.down_sampling_window = down_sampling_window

    # Optional extras commonly present in TimeMixer++ codebases
    cfg.embed = "fixed"             # keep default; not used by all variants
    cfg.freq = "h"

    # --- Training hyperparameters (matches CLI) ---
    cfg.is_training = 1
    cfg.model_id = dataset
    cfg.learning_rate = learning_rate
    cfg.batch_size = batch_size
    cfg.train_epochs = train_epochs
    cfg.patience = patience
    cfg.itr = 1
    cfg.des = "Exp"

    # Paths (adjust externally if needed)
    cfg.root_path = "./dataset/PEMS/"
    cfg.data_path = f"{dataset}.npz"

    # Model name filled by launcher (e.g., env var $model_name)
    cfg.model = "TimeMixerPP"

    return cfg
