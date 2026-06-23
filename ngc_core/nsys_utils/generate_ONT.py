

def generate_q_k_v(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id + 1 > layers_range[1]: continue
        res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q']]])
        res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k']]])
        res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'v']]])
    return res_NeurTp


def generate_qq_kk_vv(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id+1,'q']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'],[l_id+1,'k']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'v'],[l_id+1,'v']]])
    return res_NeurTp


def generate_qqq_kk_vv(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 3 == 0 and l_id + 3 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'], [l_id+1,'q'],[l_id+2,'q']]])
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'],[l_id+1,'k']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'v'],[l_id+1,'v']]])

    if layers_range[1] % 3 == 2: res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[layers_range[1]-2,'q'],[layers_range[1]-1,'q']]])

    return res_NeurTp


def generate_qq_kkk_vv(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 3 == 0 and l_id + 3 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'], [l_id+1,'k'],[l_id+2,'k']]])
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id+1,'q']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'v'],[l_id+1,'v']]])

    if layers_range[1] % 3 == 2: res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[layers_range[1]-2,'k'],[layers_range[1]-1,'k']]])

    return res_NeurTp

def generate_qq_kk_vvv(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 3 == 0 and l_id + 3 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'v'], [l_id+1,'v'],[l_id+2,'v']]])
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id+1,'q']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'],[l_id+1,'k']]])

    if layers_range[1] % 3 == 2: res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[layers_range[1]-2,'v'],[layers_range[1]-1,'v']]])
    return res_NeurTp


def generate_qq_kv(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id+1,'q']]])
        if l_id + 1 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'],[l_id,'v']]])
    return res_NeurTp

def generate_kk_qv(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'],[l_id+1,'k']]])
        if l_id + 1 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id,'v']]])
    return res_NeurTp

def generate_vv_qk(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'v'],[l_id+1,'v']]])
        if l_id + 1 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id,'k']]])
    return res_NeurTp


def generate_qkv(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id + 1 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id,'k'],[l_id,'v']]])
    return res_NeurTp


def generate_qqq_kkk_vvv(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 3 == 0 and l_id + 3 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id+1,'q'],[l_id+2,'q']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'],[l_id+1,'k'],[l_id+2,'k']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'v'],[l_id+1,'v'],[l_id+2,'v']]])
    
    if layers_range[1] % 3 == 2: 
        res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[layers_range[1]-2,'q'],[layers_range[1]-1,'q']]])
        res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[layers_range[1]-2,'k'],[layers_range[1]-1,'k']]])
        res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[layers_range[1]-2,'v'],[layers_range[1]-1,'v']]])

    return res_NeurTp


def generate_qkk_qvv(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id,'k'],[l_id+1,'k']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id+1,'q'],[l_id,'v'],[l_id+1,'v']]])
    return res_NeurTp


def generate_qvv_qkk(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id,'v'],[l_id+1,'v']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id+1,'q'],[l_id,'k'],[l_id+1,'k']]])
    return res_NeurTp


def generate_qq_kkvv(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id+1,'q']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'],[l_id+1,'k'],[l_id,'v'],[l_id+1,'v']]])
    return res_NeurTp


def generate_kk_qqvv(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'],[l_id+1,'k']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id+1,'q'],[l_id,'v'],[l_id+1,'v']]])
    return res_NeurTp


def generate_vv_qqkk(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'v'],[l_id+1,'v']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id+1,'q'],[l_id,'k'],[l_id+1,'k']]])
    return res_NeurTp



def generate_vv_qqkk(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'v'],[l_id+1,'v']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id+1,'q'],[l_id,'k'],[l_id+1,'k']]])
    return res_NeurTp


def generate_hybrid1(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 4 == 0 and l_id + 4 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'],[l_id+1,'k']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id,'v']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id+1,'q'],[l_id+1,'v']]])

            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id+2,'q'],[l_id+3,'q']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id+2,'k'],[l_id+3,'k']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id+2,'v'],[l_id+3,'v']]])
    return res_NeurTp


def generate_hybrid2(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 4 == 0 and l_id + 4 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id+1,'q']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'],[l_id+1,'k']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'v'],[l_id+1,'v']]])

            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id+2,'k'],[l_id+3,'k']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id+2,'q'],[l_id+2,'v']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id+3,'q'],[l_id+3,'v']]])

    return res_NeurTp


def generate_hybrid3(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id >= layers_range[1]//2 and l_id+2 <= layers_range[1]:
            if l_id % 2 == 0:
                res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id+1,'q']]])
                res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'],[l_id+1,'k']]])
                res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'v'],[l_id+1,'v']]])
        else:
            if l_id % 2 == 0 and l_id + 2 <= layers_range[1]//2: 
                res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'],[l_id+1,'k']]])
            if l_id + 1 <= layers_range[1]//2:
                res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id,'v']]])

    return res_NeurTp


def generate_hybrid4(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id >= layers_range[1]//2 and l_id+2 <= layers_range[1]:
            if l_id % 2 == 0 and l_id + 2 <= layers_range[1]: 
                res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'],[l_id+1,'k']]])
            if l_id + 1 <= layers_range[1]:
                res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id,'v']]])
        else:
            if l_id % 2 == 0 and l_id + 2 <= layers_range[1]//2:
                res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id+1,'q']]])
                res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'],[l_id+1,'k']]])
                res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'v'],[l_id+1,'v']]])

    return res_NeurTp


def generate_hybrid5(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 3 == 0 and l_id+3 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'k'],[l_id+1,'k'],[l_id+2,'k']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'q'],[l_id+1,'q']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id,'v'],[l_id+1,'v']]])
            res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[l_id+2,'q'],[l_id+2,'v']]])

    if layers_range[1] % 3 == 2: 
        res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[layers_range[1]-2,'q'],[layers_range[1]-1,'q']]])
        res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[layers_range[1]-2,'k'],[layers_range[1]-1,'k']]])
        res_NeurTp.append([f"model.layers.{_layerId}.self_attn.{attn_type}_proj" for (_layerId, attn_type) in [[layers_range[1]-2,'v'],[layers_range[1]-1,'v']]])

    return res_NeurTp


def generate_u_g_d(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id + 1 > layers_range[1]: continue
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up']]])
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'gate']]])
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'down']]])
    return res_NeurTp


def generate_ugd(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id + 1 > layers_range[1]: continue
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up'],[l_id,'gate'],[l_id,'down']]])
    return res_NeurTp


def generate_uu_gg_dd(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up'],[l_id+1,'up']]])
            res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'gate'],[l_id+1,'gate']]])
            res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'down'],[l_id+1,'down']]])
    return res_NeurTp


def generate_ug_d(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up'],[l_id,'gate']]])
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'down']]])
    return res_NeurTp


def generate_ug_d(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up'],[l_id,'gate']]])
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'down']]])
    return res_NeurTp


def generate_ud_g(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up'],[l_id,'down']]])
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'gate']]])
    return res_NeurTp


def generate_gd_u(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'gate'],[l_id,'down']]])
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up']]])
    return res_NeurTp


def generate_ug_dd(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up'],[l_id,'gate']]])
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'down'],[l_id+1,'down']]])
    return res_NeurTp


def generate_ud_gg(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up'],[l_id,'down']]])
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'gate'],[l_id+1,'gate']]])
    return res_NeurTp


def generate_gd_uu(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'gate'],[l_id,'down']]])
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up'],[l_id+1,'up']]])
    return res_NeurTp


def generate_u_gg_dd(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up']]])
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'gate'],[l_id+1,'gate']]])
            res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'down'],[l_id+1,'down']]])
    return res_NeurTp

def generate_uu_g_dd(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'gate']]])
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up'],[l_id+1,'up']]])
            res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'down'],[l_id+1,'down']]])
    return res_NeurTp

def generate_uu_gg_d(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'down']]])
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up'],[l_id+1,'up']]])
            res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'gate'],[l_id+1,'gate']]])
    return res_NeurTp


def generate_uu_g_d(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'gate']]])
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'down']]])
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up'],[l_id+1,'up']]])
    return res_NeurTp

def generate_u_gg_d(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up']]])
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'down']]])
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'gate'],[l_id+1,'gate']]])
    return res_NeurTp

def generate_u_g_dd(layers_range):
    res_NeurTp = []
    for l_id in range(layers_range[0], layers_range[1], 1):
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'up']]])
        res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'gate']]])
        if l_id % 2 == 0 and l_id + 2 <= layers_range[1]:
            res_NeurTp.append([f"model.layers.{_layerId}.mlp.{mlp_type}_proj" for (_layerId, mlp_type) in [[l_id,'down'],[l_id+1,'down']]])
    return res_NeurTp


def generate_NeurTp(_nt, layers_range):

    if _nt == 'q-k-v': return generate_q_k_v(layers_range)
    elif _nt == 'qq-kk-vv': return generate_qq_kk_vv(layers_range)
    elif _nt == 'qqq-kk-vv': return generate_qqq_kk_vv(layers_range)
    elif _nt == 'qq-kkk-vv': return generate_qq_kkk_vv(layers_range)
    elif _nt == 'qq-kk-vvv': return generate_qq_kk_vvv(layers_range)
    elif _nt == 'qq-kv': return generate_qq_kv(layers_range)
    elif _nt == 'kk-qv': return generate_kk_qv(layers_range)
    elif _nt == 'vv-qk': return generate_vv_qk(layers_range)
    elif _nt == 'qkv': return generate_qkv(layers_range)
    elif _nt == 'qqq-kkk-vvv': return generate_qqq_kkk_vvv(layers_range)
    elif _nt == 'qkk-qvv': return generate_qkk_qvv(layers_range)
    elif _nt == 'qvv-qkk': return generate_qvv_qkk(layers_range)
    elif _nt == 'qq-kkvv': return generate_qq_kkvv(layers_range)
    elif _nt == 'kk-qqvv': return generate_kk_qqvv(layers_range)
    elif _nt == 'vv-qqkk': return generate_vv_qqkk(layers_range)
    elif _nt == 'hybrid1': return generate_hybrid1(layers_range)
    elif _nt == 'hybrid2': return generate_hybrid2(layers_range)
    elif _nt == 'hybrid3': return generate_hybrid3(layers_range)
    elif _nt == 'hybrid4': return generate_hybrid4(layers_range)
    elif _nt == 'hybrid5': return generate_hybrid5(layers_range)
    elif _nt == 'u-g-d': return generate_u_g_d(layers_range)
    elif _nt == 'uu-gg-dd': return generate_uu_gg_dd(layers_range)
    elif _nt == 'ugd': return generate_ugd(layers_range)
    elif _nt == 'ug-d': return generate_ug_d(layers_range)
    elif _nt == 'ud-g': return generate_ud_g(layers_range)
    elif _nt == 'gd-u': return generate_gd_u(layers_range)
    elif _nt == 'ug-dd': return generate_ug_dd(layers_range)
    elif _nt == 'ud-gg': return generate_ud_gg(layers_range)
    elif _nt == 'gd-uu': return generate_gd_uu(layers_range)
    elif _nt == 'u-gg-dd': return generate_u_gg_dd(layers_range)
    elif _nt == 'uu-g-dd': return generate_uu_g_dd(layers_range)
    elif _nt == 'uu-gg-d': return generate_uu_gg_d(layers_range)
    elif _nt == 'uu-g-d': return generate_uu_g_d(layers_range)
    elif _nt == 'u-gg-d': return generate_u_gg_d(layers_range)
    elif _nt == 'u-g-dd': return generate_u_g_dd(layers_range)

    return None





