import numpy as np
import pickle
import json
import torch

def tree_map(f, tree):
    if isinstance(tree, dict):
        return {k: tree_map(f, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(f, v) for v in tree)
    elif isinstance(tree, (int, float)):
        return tree
    else:
        return f(tree)

def to_dict(tree):
    if isinstance(tree, dict):
        return {k: to_dict(v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return {i: to_dict(v) for i, v in enumerate(tree)}
    elif isinstance(tree, (int, float)):
        return tree
    else:
        return tree

def tree_map_dict(f, tree):
    return to_dict(tree_map(f, tree))
    
class DDD(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = DDD()
        return dict.__getitem__(self, key)
    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, DDD) else v for k, v in self.items()}

def dfs(torch_dict: DDD, jax_dict, prefix=None, buffs=None):
    if prefix is None:
        prefix = []
    if buffs is None:
        buffs = set()
    
    if not isinstance(jax_dict, dict):
        # leaf
        assert isinstance(jax_dict, np.ndarray), f'Expected np.ndarray at {".".join(prefix)}, got {type(jax_dict)}'

        # apply buffs
        reshape_fn = lambda x: x
        if 'flip' in buffs:
            if 'norm' in buffs:
                # do nothing
                pass
            elif 'conv' in buffs:
                assert jax_dict.ndim == 4, (prefix, jax_dict.shape)
                # from (H, W, in_ch, out_ch) to (out_ch, in_ch, H, W)
                reshape_fn = lambda x: np.transpose(x, (3, 2, 0, 1))
            else:
                # general linear layer
                assert jax_dict.ndim == 2, (prefix, jax_dict.shape)
                reshape_fn = lambda x: x.T
            
        torch_dict['.'.join(prefix)] = torch.from_numpy(reshape_fn(jax_dict))
        return
    
    for k, v in jax_dict.items():
        new_prefix = prefix.copy()
        new_buffs = buffs.copy()
        
        # rename some symbols
        RENAME_RULES = {
            'k_embedder': 'omega_embedder',
            'cfg_start_cond_tokens': 't_min_tokens',
            'cfg_end_cond_tokens': 't_max_tokens',
            'time_cond_tokens': 'time_tokens',
            'class_cond_tokens': 'class_tokens',
            'kappa_cond_tokens': 'omega_tokens',
        }
        
        if k in RENAME_RULES:
            new_prefix.append(RENAME_RULES[k])
        elif 'layers_' in k:
            # sequential
            new_prefix.append(k.replace('layers_', ''))
        elif len(k.split('_')) >= 2 and k.split('_')[-1].isdigit():
            # sequential
            new_prefix.append('_'.join(k.split('_')[:-1]) + '.' + k.split('_')[-1]) # replace last _ with .
        elif k == 'scale':
            new_prefix.append('weight')
        elif k == 'kernel':
            new_prefix.append('weight')
            new_buffs.add('flip')
        elif k == 'embedding':
            new_prefix.append('weight')
        # elif k.startswith('_flax'):
            # skip this
            # pass
        else:
            new_prefix.append(k)
        
        # special rules
        if k == 'x_embedder':
            new_buffs.add('conv')
        elif 'norm' in k:
            new_buffs.add('norm')
            
        dfs(torch_dict, v, new_prefix, new_buffs)

def params_jax_to_torch(jax_params):
    torch_dict = DDD()
    dfs(torch_dict, jax_params)
    return torch_dict.to_dict()

def load_all_from_jax_ckpt(torch_state, jax_state, torch_state_dict_order: list[str] = None, param_only=False, strict=True):
    if torch_state_dict_order is None:
        print('\033[93mWarning: torch_state_dict_order is None, this is not recommended and may lead to error\033[0m', flush=True)
        torch_state_dict_order = list(torch_state['model'].keys())
        
    jax_params = jax_state['params']
    torch_params = params_jax_to_torch(jax_params)
    print('num loaded params:', sum(p.numel() for p in torch_params.values()))
    
    torch_orig_shapes = {k: v.shape for k, v in torch_state['model'].items()}
    if strict:
        for k, v in torch_params.items():
            assert k in torch_orig_shapes, f'Key {k} not found in torch model'
            assert v.shape == torch_orig_shapes[k], f'Shape mismatch at {k}: jax {v.shape}, torch {torch_orig_shapes[k]}'
        for k in torch_orig_shapes.keys():
            assert k in torch_params, f'Key {k} not found in loaded model'
        print('\033[92mAll parameter shape checks passed successfully\033[0m', flush=True)
    else:
        new_torch_params = {}
        for k, v in torch_params.items():
            if k not in torch_orig_shapes:
                print(f'\033[93mWarning: Key {k} not found in torch model, skipping\033[0m', flush=True)
            elif v.shape != torch_orig_shapes[k]:
                print(f'\033[93mWarning: Shape mismatch at {k}: jax {v.shape}, torch {torch_orig_shapes[k]}\033[0m', flush=True)
                new_torch_params[k] = torch_state['model'][k]
            else:
                new_torch_params[k] = v
        for k in torch_orig_shapes.keys():
            if k not in torch_params:
                print(f'\033[93mWarning: Key {k} not found in loaded model, skipping\033[0m', flush=True)
                new_torch_params[k] = torch_state['model'][k]
        torch_params = new_torch_params
    
    torch_state['model'] = torch_params

    return

    # load step
    torch_state['step'] = jax_state['step']
    
    # only optimizer remains, if there is no optimizer state, skip
    if param_only or 'optimizer' not in torch_state or torch_state['optimizer'] is None:
        print('\033[93mNo optimizer state found in torch_state, skipping optimizer loading\033[0m', flush=True)
        return
    
    mu_args = params_jax_to_torch(jax_state["opt_state"]["0"]["mu"])
    nu_args = params_jax_to_torch(jax_state["opt_state"]["0"]["nu"])
    opt = torch_state['optimizer']
    assert len(opt['param_groups']) == 1, 'Only single param group supported, got length %d' % len(opt['param_groups'])
    assert len(opt['param_groups'][0]['params']) == len(torch_state_dict_order), \
        f'Optimizer param count mismatch: {len(opt["param_groups"][0]["params"])} vs {len(torch_state_dict_order)}'
        
    for i, v in opt['state'].items():
        idx = int(i)
        torch_param_name = torch_state_dict_order[idx]
        assert torch_param_name in mu_args, f'Param {torch_param_name} not found in loaded torch params'
        mu_target = mu_args[torch_param_name]
        nu_target = nu_args[torch_param_name]
        assert mu_target.shape == v['exp_avg'].shape, \
            f'Mu shape mismatch at param {torch_param_name}: optimizer {v["exp_avg"].shape} vs loaded {mu_target.shape}'
        assert nu_target.shape == v['exp_avg_sq'].shape, \
            f'Nu shape mismatch at param {torch_param_name}: optimizer {v["exp_avg_sq"].shape} vs loaded {nu_target.shape}'
        opt['state'][i]['exp_avg'] = mu_target
        opt['state'][i]['exp_avg_sq'] = nu_target
        opt['state'][i]['step'] = jax_state['opt_state']['0']['count']
    print('\033[92mOptimizer states loaded successfully\033[0m', flush=True)
    

if __name__ == '__main__':
    from flax.training import checkpoints
    import os
    from imf import iMeanFlow
    
    for size_str, path_name in {
        'MiT_B_2': 'pretrained/imf_b_640ep/checkpoint_800640',
        'MiT_L_2': 'pretrained/imf_l_640ep_fixed/checkpoint_800640',
        'MiT_M_2': 'pretrained/imf_m_640ep_fixed/checkpoint_800640',
        'MiT_XL_2': 'pretrained/imf_xl_800ep_fixed/checkpoint_1000800'
    }.items():
    
        jax_ckpt_path = os.path.abspath(path_name)
        jax_obj = checkpoints.restore_checkpoint(jax_ckpt_path, target=None)
        jax_obj = jax_obj['ema_params']
        # jax_ckpt_path = 'ema_params.pkl'
        # jax_obj = pickle.load(open(jax_ckpt_path, 'rb'))
        
        # remove 'v heads'
        jax_obj = {
            'net': {k: v for k, v in jax_obj['net'].items() if 'v_heads' not in k}
        }
        
        # json.dump(
        #     tree_map(lambda x: str(x.shape) if x is not None else None, jax_obj), 
        #     open('jax.params', 'w'),
        # )
        
        _model = iMeanFlow(size_str)
        
        
        
        prod = lambda x: np.prod(x) if len(x) > 0 else 1
        
        def _sum(t):
            return prod(t.shape) if not isinstance(t, dict) else sum(_sum(v) for v in t.values())
        print(f'Total JAX params: {_sum(jax_obj)}')
        
        # torch_shape = json.load(open('torch.params', 'r'))
        torch_shape = _model.state_dict()
        # torch_shape = 
        # print(torch_shape)
        print(f'Total Torch params: {sum(v.numel() for v in torch_shape.values())}')
        
        # torch_shape = torch_shape['model']
        # torch_loaded = params_jax_to_torch(jax_obj['params'])
        # for k, v in torch_loaded.items():
        #     assert k in torch_shape, f'Key {k} not in torch model'
        #     assert v.shape == torch_shape[k], f'Shape mismatch at {k}: jax {v.shape}, torch {torch_shape[k]}'
        # # check all keys are covered
        # for k in torch_shape.keys():
        #     assert k in torch_loaded, f'Key {k} not in loaded model'
        # print('All checks passed!')
        
        ts = {'model': torch_shape}
        load_all_from_jax_ckpt(
            torch_state=ts,
            jax_state={'params': jax_obj},
        )
        
        torch.save(ts['model'], f'{size_str}.pth')
        print(f'[Success] Saved converted model to {size_str}.pth')