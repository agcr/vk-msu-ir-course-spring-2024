import torch
from typing import OrderedDict


def filter_model(ckpt: OrderedDict, st_dict: OrderedDict) -> OrderedDict:
    filtered_ckpt = type(ckpt)()
    prefix = 'model.'
    for k, v in ckpt.items():
        if prefix in k and k[len(prefix):] in st_dict and \
            st_dict[k[len(prefix):]].shape == v.shape:
            filtered_ckpt[k[len(prefix):]] = v
    return filtered_ckpt

def load_pretrained_model(model: torch.nn.Module, ckpt_path: str, strict: bool = True):
    ckpt: OrderedDict = torch.load(ckpt_path, map_location='cpu')
    weights_st_dict = model.state_dict()
    filtered_ckpt = filter_model(ckpt['state_dict'], weights_st_dict)
    model.load_state_dict(filtered_ckpt, strict=strict)
    if not strict:
        missing_keys = []
        for k in weights_st_dict.keys():
            if k not in filtered_ckpt:
                missing_keys += [k]
        if len(missing_keys) > 0:
            out_str = ", ".join(missing_keys)
            print(f"Missing keys: {out_str}")