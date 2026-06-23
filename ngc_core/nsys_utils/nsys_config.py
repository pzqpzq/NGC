
import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Tuple, Set, Callable, List



class SharedNeuronsHybrid2(nn.Module):
    def __init__(self, target_weights_list, input_acts_list, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slice_id2pos: List[Tuple[int, int]] = []  # (row_start, row_end) per task
        self.in_dims: List[int] = []                  # original per‑task input dims

        start_row = 0
        for idx, (W, X) in enumerate(zip(target_weights_list, input_acts_list)):
            self.in_dims.append(W.size(1))
            end_row = start_row + W.size(0)
            self.slice_id2pos.append((start_row, end_row))
            start_row = end_row
        max_in_dim = max(self.in_dims)

        device = target_weights_list[0].device
        padded_W, padded_X = [], []
        for W, X, in_i in zip(target_weights_list, input_acts_list, self.in_dims):
            pad_cols = max_in_dim - in_i
            if pad_cols > 0: W, X = F.pad(W, (0, pad_cols)), F.pad(X, (0, pad_cols))
            padded_W.append(W.to(device))
            padded_X.append(X.to(device))

        X_cat, W_cat = torch.cat(padded_X, dim=0), torch.cat(padded_W, dim=0)
        U, S, Vh = torch.svd_lowrank(X_cat @ W_cat.T , q=hidden_dim, niter=5)
        A, B = U * S.unsqueeze(0), Vh

        self.input_neurons = nn.Parameter(torch.linalg.lstsq(X_cat, A)[0])   # (max_in_dim, hidden_dim)
        self.output_neurons = nn.Parameter(B)

    def slice_ouNeurons(self, slice_id):
        s, e = self.slice_id2pos[slice_id]
        return self.output_neurons[s:e]

    def slice_weight(self, slice_id: int) -> torch.Tensor:
        s, e = self.slice_id2pos[slice_id]
        W_full = (self.input_neurons @ self.output_neurons[s:e].T).T  # (out_i, max_in_dim)
        return W_full[:, :self.in_dims[slice_id]]



class SharedNeuronsHybrid3(nn.Module):
    def __init__(self, target_weights_list, input_acts_list, hidden_dim, metric_ratio=None, metric_randn=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slice_id2pos: List[Tuple[int, int]] = []  # (row_start, row_end) per task
        self.in_dims: List[int] = []                  # original per‑task input dims

        start_row = 0
        for idx, (W, X) in enumerate(zip(target_weights_list, input_acts_list)):
            self.in_dims.append(W.size(1))
            end_row = start_row + W.size(0)
            self.slice_id2pos.append((start_row, end_row))
            start_row = end_row
        max_in_dim = max(self.in_dims)

        device = target_weights_list[0].device
        padded_W, padded_X = [], []
        for W, X, in_i in zip(target_weights_list, input_acts_list, self.in_dims):
            pad_cols = max_in_dim - in_i
            if pad_cols > 0: W, X = F.pad(W, (0, pad_cols)), F.pad(X, (0, pad_cols))
            padded_W.append(W.to(device))
            padded_X.append(X.to(device))

        X_cat, W_cat = torch.cat(padded_X, dim=0), torch.cat(padded_W, dim=0)
        U, S, Vh = torch.svd_lowrank(X_cat @ W_cat.T , q=hidden_dim, niter=5)
        A, B = U * S.unsqueeze(0), Vh

        self.input_neurons = nn.Parameter(torch.linalg.lstsq(X_cat, A)[0])   # (max_in_dim, hidden_dim)
        self.output_neurons = nn.Parameter(B)
        self.metric_proj = nn.Linear(self.input_neurons.shape[1], round(self.input_neurons.shape[1]*metric_ratio))
        self.tanh = nn.Tanh()

    def slice_ouNeurons(self, slice_id):
        s, e = self.slice_id2pos[slice_id]
        return self.output_neurons[s:e]

    def slice_weight(self, slice_id: int) -> torch.Tensor:
        s, e = self.slice_id2pos[slice_id]
        expanded_inNeurons = self.tanh(self.metric_proj(self.input_neurons))
        expanded_ouNeurons = self.tanh(self.metric_proj(self.output_neurons[s:e]))
        W_full = (expanded_inNeurons @ expanded_ouNeurons.T).T  # (out_i, max_in_dim)
        return W_full[:, :self.in_dims[slice_id]]



class SharedNeuronsHybrid4(nn.Module):
    def __init__(self, target_weights_list, input_acts_list, hidden_dim, metric_ratio=None, metric_randn=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slice_id2pos: List[Tuple[int, int]] = []  # (row_start, row_end) per task
        self.in_dims: List[int] = []                  # original per‑task input dims

        start_row = 0
        for idx, (W, X) in enumerate(zip(target_weights_list, input_acts_list)):
            self.in_dims.append(W.size(1))
            end_row = start_row + W.size(0)
            self.slice_id2pos.append((start_row, end_row))
            start_row = end_row
        max_in_dim = max(self.in_dims)

        device = target_weights_list[0].device
        padded_W, padded_X = [], []
        for W, X, in_i in zip(target_weights_list, input_acts_list, self.in_dims):
            pad_cols = max_in_dim - in_i
            if pad_cols > 0: W, X = F.pad(W, (0, pad_cols)), F.pad(X, (0, pad_cols))
            padded_W.append(W.to(device))
            padded_X.append(X.to(device))

        X_cat, W_cat = torch.cat(padded_X, dim=0), torch.cat(padded_W, dim=0)
        U, S, Vh = torch.svd_lowrank(X_cat @ W_cat.T , q=hidden_dim, niter=5)
        A, B = U * S.unsqueeze(0), Vh

        self.input_neurons = nn.Parameter(torch.linalg.lstsq(X_cat, A)[0])   # (max_in_dim, hidden_dim)
        self.output_neurons = nn.Parameter(B)
        self.inN_metric_proj = nn.Linear(self.input_neurons.shape[1], round(self.input_neurons.shape[1]*metric_ratio))
        self.ouN_metric_proj = nn.Linear(self.output_neurons.shape[1], round(self.input_neurons.shape[1]*metric_ratio))
        #self.inN_metric_proj = torch.randn(self.input_neurons.shape[1], round(self.input_neurons.shape[1]*metric_ratio)).to(torch.bfloat16).to(self.input_neurons.device)
        #self.ouN_metric_proj = torch.randn(self.input_neurons.shape[1], round(self.input_neurons.shape[1]*metric_ratio)).to(torch.bfloat16).to(self.input_neurons.device)
        #self.inN_metric_proj = metric_randn[:self.input_neurons.shape[1], :round(self.input_neurons.shape[1]*metric_ratio)].to(self.input_neurons.device)
        #self.ouN_metric_proj = metric_randn[self.input_neurons.shape[1]:, round(self.input_neurons.shape[1]*metric_ratio):].to(self.input_neurons.device)
        self.tanh = nn.Tanh()

    def slice_ouNeurons(self, slice_id):
        s, e = self.slice_id2pos[slice_id]
        return self.output_neurons[s:e]

    def slice_weight(self, slice_id: int) -> torch.Tensor:
        s, e = self.slice_id2pos[slice_id]
        expanded_inNeurons = self.tanh(self.inN_metric_proj(self.input_neurons))
        expanded_ouNeurons = self.tanh(self.ouN_metric_proj(self.output_neurons[s:e]))
        #if self.inN_metric_proj.dtype != self.input_neurons.dtype: self.inN_metric_proj = self.inN_metric_proj.to(self.input_neurons.dtype)
        #if self.ouN_metric_proj.dtype != self.output_neurons.dtype: self.ouN_metric_proj = self.ouN_metric_proj.to(self.output_neurons.dtype)
        #expanded_inNeurons = self.tanh(self.input_neurons @ self.inN_metric_proj)
        #expanded_ouNeurons = self.tanh(self.output_neurons[s:e] @ self.ouN_metric_proj)
        W_full = (expanded_inNeurons @ expanded_ouNeurons.T).T  # (out_i, max_in_dim)
        return W_full[:, :self.in_dims[slice_id]]




class SharedNeuronsHybrid5(nn.Module):
    def __init__(self, target_weights_list, input_acts_list, hidden_dim, metric_ratio=None, metric_randn=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slice_id2pos: List[Tuple[int, int]] = []  # (row_start, row_end) per task
        self.in_dims: List[int] = []                  # original per‑task input dims

        start_row = 0
        for idx, (W, X) in enumerate(zip(target_weights_list, input_acts_list)):
            self.in_dims.append(W.size(1))
            end_row = start_row + W.size(0)
            self.slice_id2pos.append((start_row, end_row))
            start_row = end_row
        max_in_dim = max(self.in_dims)

        device = target_weights_list[0].device
        padded_W, padded_X = [], []
        for W, X, in_i in zip(target_weights_list, input_acts_list, self.in_dims):
            pad_cols = max_in_dim - in_i
            if pad_cols > 0: W, X = F.pad(W, (0, pad_cols)), F.pad(X, (0, pad_cols))
            padded_W.append(W.to(device))
            padded_X.append(X.to(device))

        X_cat, W_cat = torch.cat(padded_X, dim=0), torch.cat(padded_W, dim=0)
        U, S, Vh = torch.svd_lowrank(X_cat @ W_cat.T , q=hidden_dim, niter=5)
        A, B = U * S.unsqueeze(0), Vh

        self.input_neurons = nn.Parameter(torch.linalg.lstsq(X_cat, A)[0])   # (max_in_dim, hidden_dim)
        self.output_neurons = nn.Parameter(B)
        self.metric_proj = metric_randn[:self.input_neurons.shape[1], :round(self.input_neurons.shape[1]*metric_ratio)].to(torch.bfloat16).to(self.input_neurons.device)
        self.tanh = nn.Tanh()

    def slice_ouNeurons(self, slice_id):
        s, e = self.slice_id2pos[slice_id]
        return self.output_neurons[s:e]

    def slice_weight(self, slice_id: int) -> torch.Tensor:
        s, e = self.slice_id2pos[slice_id]
        #expanded_inNeurons = self.tanh(self.metric_proj(self.input_neurons))
        #expanded_ouNeurons = self.tanh(self.metric_proj(self.output_neurons[s:e]))
        if self.metric_proj.dtype != self.input_neurons.dtype: self.metric_proj = self.metric_proj.to(self.input_neurons.dtype)
        #print(self.input_neurons.dtype, self.metric_proj.dtype)
        expanded_inNeurons = self.tanh(self.input_neurons @ self.metric_proj)
        expanded_ouNeurons = self.tanh(self.output_neurons[s:e] @ self.metric_proj)
        W_full = (expanded_inNeurons @ expanded_ouNeurons.T).T  # (out_i, max_in_dim)

        return W_full[:, :self.in_dims[slice_id]]




class SharedNeuronsHybrid6(nn.Module):
    def __init__(self, target_weights_list, input_acts_list, hidden_dim, metric_ratio=None, metric_randn=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slice_id2pos: List[Tuple[int, int]] = []  # (row_start, row_end) per task
        self.in_dims: List[int] = []                  # original per‑task input dims

        start_row = 0
        for idx, (W, X) in enumerate(zip(target_weights_list, input_acts_list)):
            self.in_dims.append(W.size(1))
            end_row = start_row + W.size(0)
            self.slice_id2pos.append((start_row, end_row))
            start_row = end_row
        max_in_dim = max(self.in_dims)

        device = target_weights_list[0].device
        padded_W, padded_X = [], []
        for W, X, in_i in zip(target_weights_list, input_acts_list, self.in_dims):
            pad_cols = max_in_dim - in_i
            if pad_cols > 0: W, X = F.pad(W, (0, pad_cols)), F.pad(X, (0, pad_cols))
            padded_W.append(W.to(device))
            padded_X.append(X.to(device))

        X_cat, W_cat = torch.cat(padded_X, dim=0), torch.cat(padded_W, dim=0)
        U, S, Vh = torch.svd_lowrank(X_cat @ W_cat.T , q=hidden_dim, niter=5)
        A, B = U * S.unsqueeze(0), Vh

        self.input_neurons = nn.Parameter(torch.linalg.lstsq(X_cat, A)[0])   # (max_in_dim, hidden_dim)
        self.output_neurons = nn.Parameter(B)
        self.tanh = nn.Tanh()

    def slice_ouNeurons(self, slice_id):
        s, e = self.slice_id2pos[slice_id]
        return self.output_neurons[s:e]

    def slice_weight(self, slice_id: int) -> torch.Tensor:
        s, e = self.slice_id2pos[slice_id]
        W_d0 = self.input_neurons @ self.output_neurons[s:e].T
        W_d2 = self.input_neurons[:,2:] @ self.output_neurons[s:e][:,:-2].T
        W_d4 = self.input_neurons[:,4:] @ self.output_neurons[s:e][:,:-4].T
        W_full = (0.5*W_d0 + 0.3*self.tanh(W_d2) + 0.2*self.tanh(W_d4)).T
        return W_full[:, :self.in_dims[slice_id]]





class LinearFromSharedNSys(nn.Module):
    def __init__(self, shared_nSys, slice_id):
        super().__init__()
        self.shared = shared_nSys.to(torch.bfloat16)
        self.slice_id = slice_id

    def get_weight(self):
        return self.shared.slice_weight(self.slice_id)
    
    def forward(self, x):
        w = self.get_weight()
        if x.device != w.device: w = w.to(x.device)
        return x @ w.T
        

def map_func(storage, loc):
    # Remaps any 'cuda:X' location to 'cuda:0' (the first available GPU)
    return storage.cuda()

def load_nsys_configs(_dir, only_wYerr=False):
    loaded_nsys_records, loaded_wYerr_records = {}, {}
    wtp_records = {}
    for _filename in os.listdir(_dir):
        nt_name, raw_wYerrs = _filename.split('#')
        if not only_wYerr:
            cuda_avail = torch.cuda.is_available()
            if not cuda_avail: print('----- CUDA-insufficient:', _filename); break
            loaded_nsys_records[nt_name] = torch.load(f"{_dir}{_filename}", map_location=map_func, weights_only=False)
            for w_id, w_name in enumerate(nt_name.split('-')): loaded_wYerr_records[w_name] = float(raw_wYerrs.replace('.pt','').split(';')[w_id])
        if only_wYerr:
            w_errs = [float(raw_wYerrs.replace('.pt','').split(';')[w_id]) for w_id, w_name in enumerate(nt_name.split('-'))]
            wtp_records[nt_name] = w_errs

    if not only_wYerr: return loaded_nsys_records, loaded_wYerr_records
    else: return wtp_records



def replace_Linear_with_nSys(cur_module, nsys_records, wYerr_records, _prefix = ""):

    mean_Yerr = sum([wYerr_records[_wN] for _wN in wYerr_records])/len(wYerr_records)
    for _name, _child in cur_module.named_children():
        cur_name = f"{_prefix}{_name}"
        if isinstance(_child, nn.Linear):# and cur_name in attn_weight_names + mlp_weight_names:
            for OST_key in nsys_records.keys():
                if cur_name in OST_key:
                    _slice_id = OST_key.split('-').index(cur_name)
                    _Yerr = wYerr_records[cur_name]
                    if _Yerr < mean_Yerr*1.5: setattr(cur_module, _name, LinearFromSharedNSys(nsys_records[OST_key], _slice_id))
                    break
        else: replace_Linear_with_nSys(_child, nsys_records, wYerr_records, _prefix = cur_name + ".")
    return None



def replace_Linear_with_nSys_v2(cur_module, nsys_records, _prefix = ""):

    #mean_Yerr = sum([wYerr_records[_wN] for _wN in wYerr_records])/len(wYerr_records)
    for _name, _child in cur_module.named_children():
        cur_name = f"{_prefix}{_name}"
        if isinstance(_child, nn.Linear):# and cur_name in attn_weight_names + mlp_weight_names:
            for OST_key in nsys_records.keys():
                if cur_name in OST_key:
                    _slice_id = OST_key.split('-').index(cur_name)
                    #_Yerr = wYerr_records[cur_name]
                    setattr(cur_module, _name, LinearFromSharedNSys(nsys_records[OST_key], _slice_id))
                    break
        else: replace_Linear_with_nSys_v2(_child, nsys_records, _prefix = cur_name + ".")
    return None


