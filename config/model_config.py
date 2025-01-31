from ml_collections import ConfigDict
import numpy as np 


def training_config():
    config = ConfigDict()

    config.train = True
    config.evaluate = True
    config.dataset = 'ECG_data'
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
    config.dropout_rate = 0.6
    config.leakyrelu_rate = 0.2

    return config


def fourier_forecast_config():
    config = ConfigDict()
    config.n_layers = 2
    config.map_layer_num = 3
    config.n_cluster_facs = [1.0, 0.5] #, 0.5]#, 0.25]
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
    config.lifter_image_embed = True 
    config.embed_domain_embed = True 

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