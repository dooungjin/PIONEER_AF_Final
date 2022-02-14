import os
import torch
import pkg_resources
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from pioneer.pioneer_nn.models import *


def seq_seq_prediction(seq_seq_data, out_dir, device):
    """
    Make PIONEER interface predictions with the sequence-sequence model. Predictions will be written in `out_dir`
    in *.npy format and can be loaded with np.load(, allow_pickle=True).
    
    Args:
        seq_seq_data (list): List of dictionaries generated by `generate_seq_seq_data` in `data_generation.py`.
        out_dir (str): Path to the directory to store predictions.
        device (str): On the new server, should be one of 'cuda:0', 'cuda:1' or 'cpu'.
    
    Returns:
        None.
    
    """
    if not seq_seq_data:
        return
    unit = 'GRU'
    output_size, hidden_dim = 2, 256
    n_layers, batch_size, drop_prob = 2, 1, 0.5
    input_dim = seq_seq_data[0]['p1_feature'].shape[1]
    model = RNN_NET(unit, input_dim, output_size, hidden_dim, n_layers, drop_prob)
    model.load_state_dict(torch.load(pkg_resources.resource_filename(__name__, 'data/saved_models/seq_seq.pt'), map_location='cpu'))
    model.to(device)
    h = model.init_hidden(unit, batch_size)
    
    model.eval()
    # Predict for p1 for all interactions
    for idx, data in enumerate(seq_seq_data):
        if os.path.exists(os.path.join(out_dir, data['complex'] + '_0.npy')):
            continue

        # Input shape should be (batch, seq_len, input_dim)
        p1_feature = np.expand_dims(data['p1_feature'], axis=0)
        p2_feature = np.expand_dims(data['p2_feature'], axis=0)
        p1_feature = torch.Tensor(p1_feature).to(device)
        p2_feature = torch.Tensor(p2_feature).to(device)
        
        res_sample = np.asarray(data['p1_sample'])
        y_pred = model(unit, p1_feature, p2_feature, res_sample, h)
        prob = softmax(y_pred.cpu().detach().numpy(), axis=1)[:,1:]
        pred = np.concatenate((prob, np.expand_dims(res_sample, axis=1)), axis=1)
        pred = np.array(sorted(pred, key=lambda a_entry: a_entry[1]))
        pred[:,1] = (pred[:,1] + 1)
        prob, res_id = pred[:,0], pred[:,1]
        res_id = res_id.astype(int)
        mod = np.empty(res_id.shape[0], dtype='object')
        mod[:] = 'seq_seq'
        
        res_id = np.expand_dims(res_id, axis=1)
        prob = np.expand_dims(prob, axis=1)
        mod = np.expand_dims(mod, axis=1)
        
        prediction = np.concatenate((res_id, prob, mod), axis=1)
        np.save(os.path.join(out_dir, data['complex'] + '_0.npy'), prediction)
    
    # Predict for p2 for only heterodimers
    for idx, data in enumerate(seq_seq_data):
        p1, p2 = data['complex'].split('_')
        if p1 == p2:
            continue

        if os.path.exists(os.path.join(out_dir, data['complex'] + '_1.npy')):
            continue

        p1_feature = np.expand_dims(data['p2_feature'], axis=0)
        p2_feature = np.expand_dims(data['p1_feature'], axis=0)
        p1_feature = torch.Tensor(p1_feature).to(device)
        p2_feature = torch.Tensor(p2_feature).to(device)
        
        res_sample = np.asarray(data['p2_sample'])
        y_pred = model(unit, p1_feature, p2_feature, res_sample, h)
        prob = softmax(y_pred.cpu().detach().numpy(), axis=1)[:,1:]
        pred = np.concatenate((prob, np.expand_dims(res_sample, axis=1)), axis=1)
        pred = np.array(sorted(pred, key=lambda a_entry: a_entry[1]))
        pred[:,1] = (pred[:,1] + 1)
        prob, res_id = pred[:,0], pred[:,1]
        res_id = res_id.astype(int)
        mod = np.empty(res_id.shape[0], dtype='object')
        mod[:] = 'seq_seq'
        
        res_id = np.expand_dims(res_id, axis=1)
        prob = np.expand_dims(prob, axis=1)
        mod = np.expand_dims(mod, axis=1)
        
        prediction = np.concatenate((res_id, prob, mod), axis=1)
        np.save(os.path.join(out_dir, data['complex'] + '_1.npy'), prediction)
        
        
def seq_str_prediction(seq_str_data, out_dir, device):
    """
    Make PIONEER interface predictions with the sequence-structure model. Predictions will be written in `out_dir`
    in *.npy format and can be loaded with np.load(, allow_pickle=True).
    
    Args:
        seq_str_data (list): List of dictionaries generated by `generate_seq_str_data` in `data_generation.py`.
        out_dir (str): Path to the directory to store predictions.
        device (str): On the new server, should be one of 'cuda:0', 'cuda:1' or 'cpu'.
    
    Returns:
        None.
    
    """
    if not seq_str_data:
        return
    drop_prob = 0
    gcn_input_dim = seq_str_data[0]['partner_gcn_feature'].shape[1]
    rnn_input_dim = seq_str_data[0]['partner_rnn_feature'].shape[1]
    str_model = SEQ_STR_NET2(gcn_input_dim, rnn_input_dim)
    str_model.load_state_dict(torch.load(pkg_resources.resource_filename(__name__, 'data/saved_models/str_str.pt'), map_location='cpu'))
    str_model.to(device)
    str_h = str_model.init_hidden(1)
    
    input_dim = seq_str_data[0]['rnn_feature'].shape[1]
    seq_model = SEQ_STR_NET1('GRU', input_dim)
    seq_model.load_state_dict(torch.load(pkg_resources.resource_filename(__name__, 'data/saved_models/seq_seq.pt'), map_location='cpu'))
    seq_model.to(device)
    seq_h = seq_model.init_hidden('GRU', 1)
    
    model = SEQ_STR_NET(drop_prob)
    model.load_state_dict(torch.load(pkg_resources.resource_filename(__name__, 'data/saved_models/seq_str.pt'), map_location='cpu'))
    model.to(device)
    
    str_model.eval()
    seq_model.eval()
    model.eval()
    for data in seq_str_data:
        rnn_feature = data['rnn_feature']
        rnn_feature = np.expand_dims(rnn_feature, axis=0)
        rnn_feature = torch.Tensor(rnn_feature).to(device)
        
        partner_gcn_feature = torch.Tensor(data['partner_gcn_feature']).to(device)
        partner_edge_index = torch.LongTensor(data['partner_edge_index']).to(device)
        
        partner_rnn_feature = data['partner_rnn_feature']
        partner_rnn_feature = np.expand_dims(partner_rnn_feature, axis=0)
        partner_rnn_feature = torch.Tensor(partner_rnn_feature).to(device)
        
        # Residues covered by PDB
        partner_res_id = np.array(data['partner_res_id'])
        res_sample = np.asarray(data['sample'])
        
        seq_emb = seq_model('GRU', rnn_feature, seq_h)
        str_emb = str_model(partner_gcn_feature, partner_edge_index, partner_rnn_feature, str_h, partner_res_id)
        y_pred = model(res_sample, seq_emb, str_emb)
        prob = softmax(y_pred.cpu().detach().numpy(), axis=1)[:,1:]
        pred = np.concatenate((prob, np.expand_dims(res_sample, axis=1)), axis=1)
        pred = np.array(sorted(pred, key=lambda a_entry: a_entry[1]))
        pred[:,1] = (pred[:,1] + 1)
        prob, res_id = pred[:,0], pred[:,1]
        res_id = res_id.astype(int)
        mod = np.empty(res_id.shape[0], dtype='object')
        mod[:] = 'seq_str'
        
        res_id = np.expand_dims(res_id, axis=1)
        prob = np.expand_dims(prob, axis=1)
        mod = np.expand_dims(mod, axis=1)
        
        prediction = np.concatenate((res_id, prob, mod), axis=1)
        np.save(os.path.join(out_dir, data['complex'] + '.npy'), prediction)


def str_seq_prediction(str_seq_data, out_dir, device):
    """
    Make PIONEER interface predictions with the structure-sequence model. Predictions will be written in `out_dir`
    in *.npy format and can be loaded with np.load(, allow_pickle=True).
    
    Args:
        str_seq_data (list): List of dictionaries generated by `generate_str_seq_data` in `data_generation.py`.
        out_dir (str): Path to the directory to store predictions.
        device (str): On the new server, should be one of 'cuda:0', 'cuda:1' or 'cpu'.
    
    Returns:
        None.
    
    """
    if not str_seq_data:
        return
    drop_prob = 0
    gcn_input_dim = str_seq_data[0]['gcn_feature'].shape[1]
    rnn_input_dim = str_seq_data[0]['rnn_feature'].shape[1]
    str_model = STR_SEQ_NET1(gcn_input_dim, rnn_input_dim)
    str_model.load_state_dict(torch.load(pkg_resources.resource_filename(__name__, 'data/saved_models/str_str.pt'), map_location='cpu'))
    str_model.to(device)
    str_h = str_model.init_hidden(1)
    
    # Defining and loading pre-trained RNN model
    input_dim = str_seq_data[0]['partner_rnn_feature'].shape[1]
    seq_model = STR_SEQ_NET2('GRU', input_dim)
    seq_model.load_state_dict(torch.load(pkg_resources.resource_filename(__name__, 'data/saved_models/seq_seq.pt'), map_location='cpu'))
    seq_model.to(device)
    seq_h = seq_model.init_hidden('GRU', 1)
    
    model = STR_SEQ_NET(drop_prob)
    model.load_state_dict(torch.load(pkg_resources.resource_filename(__name__, 'data/saved_models/str_seq.pt'), map_location='cpu'))
    model.to(device)
    
    str_model.eval()
    seq_model.eval()
    model.eval()
    for data in str_seq_data:
        gcn_feature = torch.Tensor(data['gcn_feature']).to(device)
        edge_index = torch.LongTensor(data['edge_index']).to(device)
        
        rnn_feature = data['rnn_feature']
        rnn_feature = np.expand_dims(rnn_feature, axis=0)
        rnn_feature = torch.Tensor(rnn_feature).to(device)
        
        # Residues covered by PDB
        res_id = np.array(data['res_id'])
        res_sample = np.asarray(data['sample'])
        uncovered = np.asarray([np.where(res_id == res_sample[j])[0][0] for j in range(len(res_sample))])
        
        partner_feature = data['partner_rnn_feature']
        partner_feature = np.expand_dims(partner_feature, axis = 0)
        partner_feature = torch.Tensor(partner_feature).to(device)
        
        str_emb = str_model(gcn_feature, edge_index, rnn_feature, str_h, res_id)
        seq_emb = seq_model('GRU', partner_feature, seq_h)
        y_pred = model(uncovered, str_emb, seq_emb)
        prob = softmax(y_pred.cpu().detach().numpy(), axis=1)[:,1:]
        res_id = (res_sample + 1).astype(int)
        mod = np.empty(res_id.shape[0], dtype='object')
        mod[:] = 'str_seq'
        res_id = np.expand_dims(res_id, axis=1)
        mod = np.expand_dims(mod, axis=1)
        
        prediction = np.concatenate((res_id, prob, mod), axis=1)
        prediction = np.array(sorted(prediction, key=lambda a_entry: a_entry[0]))
        np.save(os.path.join(out_dir, data['complex'] + '.npy'), prediction)
        

def str_str_prediction(str_str_data, out_dir, device):
    """
    Make PIONEER interface predictions with the structure-structure model. Predictions will be written in `out_dir`
    in *.npy format and can be loaded with np.load(, allow_pickle=True).
    
    Args:
        str_str_data (list): List of dictionaries generated by `generate_str_str_data` in `data_generation.py`.
        out_dir (str): Path to the directory to store predictions.
        device (str): On the new server, should be one of 'cuda:0', 'cuda:1' or 'cpu'.
    
    Returns:
        None.
    
    """
    if not str_str_data:
        return
    gcn_input_dim = str_str_data[0]['gcn_feature'].shape[1]
    rnn_input_dim = str_str_data[0]['rnn_feature'].shape[1]
    gcn_out = 128
    gcn_layers = 2
    rnn_hidden = 128
    rnn_layers = 2
    rnn_dropout = 0.0
    dropout = 0.0
    
    model = GCN_RNN_NET(gcn_input_dim, gcn_out, gcn_layers, 0, rnn_input_dim, rnn_hidden, rnn_layers, rnn_dropout, 1, dropout, 2, 'gaussian', 150, 20)
    model.load_state_dict(torch.load(pkg_resources.resource_filename(__name__, 'data/saved_models/str_str.pt'), map_location='cpu'), strict=False)
    model.to(device)
    h = model.init_hidden(1)
    
    model.eval()
    for data in str_str_data:
        gcn_feature = torch.Tensor(data['gcn_feature']).to(device)
        edge_index = torch.LongTensor(data['edge_index']).to(device)
        partner_gcn_feature = torch.Tensor(data['partner_gcn_feature']).to(device)
        partner_edge_index = torch.LongTensor(data['partner_edge_index']).to(device)
        rnn_feature = data['rnn_feature']
        rnn_feature = np.expand_dims(rnn_feature, axis=0)
        partner_rnn_feature = data['partner_rnn_feature']
        partner_rnn_feature = np.expand_dims(partner_rnn_feature, axis=0)
        rnn_feature = torch.Tensor(rnn_feature).to(device)
        partner_rnn_feature = torch.Tensor(partner_rnn_feature).to(device)
        
        # All residue ids covered by PDB or ModBase
        res_id = np.array(data['res_id'])
        partner_res_id = np.array(data['partner_res_id'])
        
        # Residue ids that will be predicted
        res_sample = np.asarray(data['sample'])
        # Indices of residue ids that will be predicted. These indices will be used for `gcn_feature` that is a subset of all features.
        uncovered = np.asarray([np.where(res_id == res_sample[j])[0][0] for j in range(len(res_sample))])
        y_pred = model(gcn_feature, edge_index, partner_gcn_feature, partner_edge_index, uncovered, rnn_feature, partner_rnn_feature, h, res_id, partner_res_id)
        prob = softmax(y_pred.cpu().detach().numpy(), axis=1)[:,1:]
        pred = np.concatenate((prob, np.expand_dims(res_sample, axis=1)), axis = 1)
        pred = np.array(sorted(pred, key=lambda a_entry: a_entry[1]))
        pred[:,1] = (pred[:,1] + 1)
        prob, res_id = pred[:,0], pred[:,1]
        res_id = res_id.astype(int)
        mod = np.empty(res_id.shape[0], dtype='object')
        mod[:] = 'str_str'
        
        res_id = np.expand_dims(res_id, axis=1)
        prob = np.expand_dims(prob, axis=1)
        mod = np.expand_dims(mod, axis=1)
        
        prediction = np.concatenate((res_id, prob, mod), axis = 1)
        np.save(os.path.join(out_dir, data['complex'] + '.npy'), prediction)
