import json
from datetime import datetime

from StemGNN.data_loader.forecast_dataloader import ForecastDataset, de_normalized

import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np
import time
import os

from math_utils import evaluate


def masked_MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mask = (v == 0)
    percentage = np.abs(v_ - v) / np.abs(v)
    if np.any(mask):
        masked_array = np.ma.masked_array(percentage, mask=mask)  # mask the dividing-zero as invalid
        result = masked_array.mean(axis=axis)
        if isinstance(result, np.ma.MaskedArray):
            return result.filled(np.nan)
        else:
            return result
    return np.mean(percentage, axis).astype(np.float64)


def MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mape = (np.abs(v_ - v) / np.abs(v)+1e-5).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)


def RMSE(v, v_, axis=None):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64)


def MAE(v, v_, axis=None):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v), axis).astype(np.float64)

import torch

def compute_dirichlet_energy(predictions, edge_index, edge_weight):
    """
    Compute the Dirichlet energy in a vectorized way.

    Args:
        predictions (torch.Tensor): Node feature matrix of shape (num_nodes, num_features).
        edge_index (torch.Tensor): Edge index matrix of shape (2, num_edges), where each column defines an edge (source, target).
        edge_weight (torch.Tensor): Edge weights of shape (num_edges,).

    Returns:
        torch.Tensor: The Dirichlet energy value (scalar).
    """
    num_nodes = torch.max(edge_index) + 1 
    degrees = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long, device=edge_index.device))
    
    source, target = edge_index

    degrees = degrees.cpu().numpy()
    source = source.cpu().numpy()
    target = target.cpu().numpy()

    # Compute the difference between predictions at the source and target nodes
    diff = predictions[source]/degrees[source] - predictions[target]/degrees[target]

    # Compute the squared L2 norm of the differences for each edge
    squared_norm = torch.sum(diff ** 2, dim=-1)

    # Compute the weighted sum of squared norms
    dirichlet_energy = torch.sum(edge_weight * squared_norm)

    return dirichlet_energy


def evaluate(y, y_hat, by_step=False, by_node=False, edge_index=None, edge_weights=None, last_pred=None):
    '''
    :param y: array in shape of [count, time_step, node].
    :param y_hat: in same shape with y.
    :param by_step: evaluate by time_step dim.
    :param by_node: evaluate by node dim.
    :return: array of mape, mae and rmse.
    '''

    if edge_index is not None and edge_weights is not None and False:
        edge_index, edge_weights = edge_index[-1], edge_weights[-1]
        dirchlet_engrey = torch.abs(compute_dirichlet_energy(last_pred, edge_index, edge_weights))
    else:
        dirchlet_engrey = -1.0 

    if not by_step and not by_node:
        s = MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat)
    if by_step and by_node:
        s = MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0)
    if by_step:
        s = MAPE(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2))
    if by_node:
        s = MAPE(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1))

    return *s, dirchlet_engrey


def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model


def inference(model, dataloader, device, node_cnt, window_size, horizon):
    forecast_set = []
    target_set = []
    model.eval()
    #node_cnt -= 4 
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)

            inputs = inputs.float()
            target = target.float()
            #x_enc, x_enc_mark, y_enc, y_enc_mark = inputs[:,:,:-4], inputs[:,:,-4:], target[:,:,:-4], target[:,:,-4:]

            step = 0
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float64)
            while step < horizon:
                #dec_inp = torch.zeros_like(y_enc[:, -3:, :]).float()
                #dec_inp = torch.cat([y_enc[:, :12, :], dec_inp], dim=1).float().to(device)
                #forecast_result = model(x_enc, x_enc_mark, dec_inp, y_enc_mark)
                forecast_result = model(inputs)
                len_model_output = forecast_result.size()[1]
                if len_model_output == 0:  
                    raise Exception('Get blank inference result')

                #target = target[:,-3:,:-4]

                target = target[:,:,:forecast_result.shape[-1]]
                inputs = inputs[:,:,:forecast_result.shape[-1]]

                inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,:].clone()
                inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
                step += min(horizon - step, len_model_output)
            forecast_set.append(forecast_steps)
            target_set.append(target.detach().cpu().numpy())
    
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)


def validate(model, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon,
             result_file=None):
    
    start = datetime.now()
    forecast_norm, target_norm = inference(model, dataloader, device,
                                           node_cnt, window_size, horizon)   
    if normalize_method and statistic and False:
        forecast = de_normalized(forecast_norm, normalize_method, statistic) 
        target = de_normalized(target_norm, normalize_method, statistic)
    else:
        forecast, target = forecast_norm, target_norm

    if hasattr(model, 'edge_indexs'):   
        score = evaluate(target, forecast, edge_index=model.edge_indexs, edge_weights=model.edge_weights, last_pred=model.last_pred)
    else:
        score = evaluate(target, forecast)
    
    score_by_node = evaluate(target, forecast, by_node=True)
    end = datetime.now()

    score_norm = evaluate(target_norm, forecast_norm)
    print(f'NORM: MAPE {score_norm[0]:7.9%}; MAE {score_norm[1]:7.9f}; RMSE {score_norm[2]:7.9f}.')
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
    if result_file:
        if not os.path.exists(result_file):
            os.makedirs(result_file)
        step_to_print = 0
        forcasting_2d = forecast[:, step_to_print, :]
        forcasting_2d_target = target[:, step_to_print, :]

        np.savetxt(f'{result_file}/target.csv', forcasting_2d_target, delimiter=",")
        np.savetxt(f'{result_file}/predict.csv', forcasting_2d, delimiter=",")
        np.savetxt(f'{result_file}/predict_abs_error.csv',
                   np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
        np.savetxt(f'{result_file}/predict_ape.csv',
                   np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")

    d = -1 if len(score) < 4 else score[3]
    return dict(mae=score[1], mae_node=score_by_node[1], mape=score[0], mape_node=score_by_node[0],
                rmse=score[2], rmse_node=score_by_node[2], norm_mae=score_norm[1], dirchlet_energy=d)


def train(train_data, valid_data, args, result_file, model=None, loss=nn.MSELoss()):
    node_cnt = train_data.shape[1]
    if model is None:
        model = Model(node_cnt, 2, args.window_size, args.multi_layer, horizon=args.horizon)
    model.to(args.device)

    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')

    if args.norm_method == 'z_score':
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == 'min_max':
        train_min = np.min(train_data, axis=0)
        train_max = np.max(train_data, axis=0)
        normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        normalize_statistic = None

    if normalize_statistic is not None:
        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
            json.dump(normalize_statistic, f)

    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                         num_workers=0)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    forecast_loss = loss.to(args.device)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    best_validate_mae = np.inf
    validate_score_non_decrease_count = 0
    performance_metrics = {}
    model = model.float() 
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            
            inputs = inputs.float()
            target = target.float()
            #x_enc, x_enc_mark, y_enc, y_enc_mark = inputs[:,:,:-4], inputs[:,:,-4:], target[:,:,:-4], target[:,:,-4:]

            #dec_inp = torch.zeros_like(y_enc[:, -3:, :]).float()
            #dec_inp = torch.cat([y_enc[:, :12, :], dec_inp], dim=1).float().to(args.device)

            model.zero_grad()        

            #forecast = model(x_enc, x_enc_mark, dec_inp, y_enc_mark)
            forecast = model(inputs)
            #target = target[:,-3:,:-4]

            if hasattr(model, 'all_hidden_outputs'):                                
                loss = forecast_loss(forecast, target, model.all_hidden_outputs)    
            else:
                loss = forecast_loss(forecast, target)

            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} | best mae {:5.4f}'.format(epoch, (
                time.time() - epoch_start_time), loss_total / cnt, best_validate_mae))
        save_model(model, result_file, epoch)
        if (epoch+1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
 
            performance_metrics = \
                validate(model, valid_loader, args.device, args.norm_method, normalize_statistic,
                         node_cnt, args.window_size, args.horizon,
                         result_file=result_file)
            if best_validate_mae > performance_metrics['mae']:
                best_validate_mae = performance_metrics['mae']
                is_best_for_now = True
                validate_score_non_decrease_count = 0
            else:
                validate_score_non_decrease_count += 1
            # save model
            if is_best_for_now:
                save_model(model, result_file)
        # early stop
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    return performance_metrics, normalize_statistic


def test(model, test_data, args, path, normalize_statistic):
    with open(os.path.join(path, 'norm_stat.json'),'r') as f:
       normalize_statistic = json.load(f)

    node_cnt = test_data.shape[1]
    test_set = ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                               normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)

    performance_metrics = validate(model, test_loader, args.device, args.norm_method, normalize_statistic,
                                    node_cnt, args.window_size, args.horizon,
                                    result_file=path)
    mae, mape, rmse, dirch = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse'], performance_metrics['dirchlet_energy']
    print('Performance on test set: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.4f}'.format(mape, mae, rmse))

    return mae, mape, rmse, dirch 
