

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def generate_IO_tensors(_records, weight_name, device=None):
    _inputs_act = [_records[_id][weight_name][0][0] for _id in _records]
    _outputs_act = [_records[_id][weight_name][1][0] for _id in _records]
    if device is None: return torch.cat(_inputs_act, dim=0), torch.cat(_outputs_act, dim=0)
    return torch.cat(_inputs_act, dim=0).to(device), torch.cat(_outputs_act, dim=0).to(device)



def eval_nsys_Yerr(_nsys, _NeurTp, test_IOs, tmp_device):

    wYerr_dict = {}
    with torch.no_grad():
        for w_id, _name in enumerate(_NeurTp):
            test_input, test_output = test_IOs[w_id]
            nsys_output = test_input.to(tmp_device) @ _nsys.slice_weight(w_id).T
            nsys_err = torch.mean((nsys_output - test_output.to(tmp_device))**2)/torch.var(test_output).to(tmp_device)
            #print(_name, nsys_err)
            wYerr_dict[_name] = nsys_err.item()

    torch.cuda.empty_cache()
    return wYerr_dict


def initialize_nsys(cur_NeurTp, target_weights, svd_records, test_records, comp_ratio, nsys_module, 
                    metric_ratio=None, nsys_type=None):

    metric_randn = None
    nsys_records = {}; wYerr_records = {}
    for _NeurTp in cur_NeurTp:
        tmp_device = target_weights[_NeurTp[0]].device
        svd_IOs = [generate_IO_tensors(svd_records, _name, tmp_device) for _name in _NeurTp]
        test_IOs = [generate_IO_tensors(test_records, _name, tmp_device) for _name in _NeurTp]

        shareW_list = [target_weights[_name].to(tmp_device).to(torch.float32) for _name in _NeurTp]
        inputAct_list = [svd_item[0].to(torch.float32) for svd_item in svd_IOs]

        num_rawParams = sum([_w.numel() for _w in shareW_list])
        k_share = round(comp_ratio * num_rawParams/(sum([_W.shape[0] for _W in shareW_list])+max([_W.shape[1] for _W in shareW_list])))
    
        if metric_ratio is not None: 
            tmp_nSys = nsys_module(shareW_list, inputAct_list, k_share, metric_ratio=metric_ratio, metric_randn=metric_randn).to(tmp_device).to(torch.bfloat16)
        else: tmp_nSys = nsys_module(shareW_list, inputAct_list, k_share).to(tmp_device).to(torch.bfloat16)

        cur_wYerr = eval_nsys_Yerr(tmp_nSys, _NeurTp, test_IOs, tmp_device)
        for _w in cur_wYerr:
            if _w not in wYerr_records: wYerr_records[_w] = cur_wYerr[_w]
            else: print(f"--- error occur on Weight {_w} ---")
        nsys_records['-'.join(_NeurTp)] = tmp_nSys
        
    return nsys_records, wYerr_records


def train_single_nsys(_nsys, _NeurTp, _records, shareW_list, _device, max_epochs=5, LR=2e-6):

    loss_fn = nn.MSELoss()
    _optimizer = optim.Adam(_nsys.parameters(), lr=LR)
    tmp_records = {tr_id: {w_name: [_records[tr_id][w_name][0].to(_device), _records[tr_id][w_name][1].to(_device)] for w_name in _NeurTp} for tr_id in _records}
            
    for _ep in range(max_epochs):
        W_loss = 0
        for w_id, w_name in enumerate(_NeurTp): W_loss += loss_fn(_nsys.slice_weight(w_id), shareW_list[w_id])
        _optimizer.zero_grad(); W_loss.backward(); _optimizer.step()

        for tr_id in _records:
            Y_loss = 0
            for w_id, w_name in enumerate(_NeurTp):
                _inAct, _ouAct = tmp_records[tr_id][w_name]
                Y_loss += loss_fn(_inAct.to(torch.float32) @ _nsys.slice_weight(w_id).T, _ouAct.to(torch.float32))
            _optimizer.zero_grad(); Y_loss.backward(); _optimizer.step()

    torch.cuda.empty_cache()
    return _nsys.to(torch.bfloat16)


def train_nsys_by_epochs(raw_nsys_records, cur_NeurTp, target_weights, train_records, test_records, max_epochs=100):

    nsys_records = {}; wYerr_records = {}
    for _NeurTp in cur_NeurTp:
        tmp_device = target_weights[_NeurTp[0]].device
        test_IOs = [generate_IO_tensors(test_records, _name, tmp_device) for _name in _NeurTp]
        shareW_list = [target_weights[_name].to(tmp_device).to(torch.float32) for _name in _NeurTp]
        
        tmp_nSys = raw_nsys_records['-'.join(_NeurTp)]
        tmp_nSys = train_single_nsys(tmp_nSys.to(torch.float32), _NeurTp, train_records, shareW_list, tmp_device, max_epochs=max_epochs, LR=2e-6)

        cur_wYerr = eval_nsys_Yerr(tmp_nSys, _NeurTp, test_IOs, tmp_device)
        for _w in cur_wYerr:
            if _w not in wYerr_records: wYerr_records[_w] = cur_wYerr[_w]
            else: print(f"--- error occur on Weight {_w} ---")
        nsys_records['-'.join(_NeurTp)] = tmp_nSys
        
    return nsys_records, wYerr_records

