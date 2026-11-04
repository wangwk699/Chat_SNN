# Extracted from the local PhaseSNN implementation; see NOTICE and LICENSE.
import json
from types import SimpleNamespace
import torch

def load_json_as_namespace(file_path):
    """
    Loads a JSON file from the specified path and returns its content as a SimpleNamespace.

    Parameters:
    - file_path (str): The file path of the JSON file to be loaded.

    Returns:
    - SimpleNamespace: The content of the JSON file as a SimpleNamespace.
    """
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return json.loads(json.dumps(data), object_hook=lambda d: SimpleNamespace(**d))

def replicate_past_key_values(past_key_values, T):
    """
    将原有的 past_key_values 沿时间维度复制 T 次，组成新的 [T, B, H, L, D] 结构。
    """
    new_past_kv = []
    for key, value in past_key_values:

        tmp_key = key[0].clone()/T
        tmp_value = value[0].clone()/T
        key_repeat = torch.stack([tmp_key for _ in range(T)], dim=0)
        value_repeat = torch.stack([tmp_value for _ in range(T)], dim=0)


        key_repeat = key_repeat.permute(1, 0, 2, 3, 4)
        value_repeat = value_repeat.permute(1, 0, 2, 3, 4)
        new_past_kv.append((key_repeat, value_repeat))

    return tuple(new_past_kv)
def mv_kv_cache(key_values, model=None, dev=None):
    '''
    move prefixed_key_values to corresponding device through full model or target_dec
    '''
    assert model is None or dev is None
    if key_values is None:
        return None
    key_values = list(key_values)
    if model is not None:
        layers = model.model.layers
        for layer_index in range(len(key_values)):
            block_dev = next(layers[layer_index].parameters()).device
            key_values[layer_index] = list(key_values[layer_index])
            key_values[layer_index][0] = key_values[layer_index][0].to(block_dev)
            key_values[layer_index][1] = key_values[layer_index][1].to(block_dev)
            key_values[layer_index] = tuple(key_values[layer_index])

    if dev is not None:
        for layer_index in range(len(key_values)):
            key_values[layer_index] = list(key_values[layer_index])
            key_values[layer_index][0] = key_values[layer_index][0].to(dev)
            key_values[layer_index][1] = key_values[layer_index][1].to(dev)
            key_values[layer_index] = tuple(key_values[layer_index])
    key_values = tuple(key_values)
    return key_values
