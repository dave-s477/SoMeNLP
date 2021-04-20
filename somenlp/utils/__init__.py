import torch
import argparse
import operator

from functools import reduce

from .time_marker import get_time_marker

def str2bool(v):
    """Tranform a string value into bool

    Args:
        v (str): string indicating a bool value

    Raises:
        argparse.ArgumentTypeError: str is unsuited for parsing to boolean

    Returns:
        bool: result
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_dropout(model, drop_rate=0.3):
    """Recursively set dropout in pytorch model

    Args:
        model (pytorch model): model
        drop_rate (float, optional): value for dropouts. Defaults to 0.3.
    """
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)    

# TODO recursive would be nicer.. 
def find_type_in_dict(dictionary, data_type=list):
    """Get all entries in a dictionary that are of a certain type up to depth 3

    Args:
        dictionary (dict): input dictionary
        data_type (python data type identifier, optional): data type to match. Defaults to list.

    Returns:
        list: paths of matched elements within the dictionary
    """
    paths_to_examine = []
    for main_key, main_values in dictionary.items():
        current_path = [main_key]
        if isinstance(main_values, data_type):
            paths_to_examine.append({
                'path': current_path,
                'values': main_values
            })
        elif isinstance(main_values, dict):
            for sub_key, sub_values in main_values.items():
                current_path = [main_key, sub_key]
                if isinstance(sub_values, data_type):
                    paths_to_examine.append({
                        'path': current_path,
                        'values': sub_values
                    })
                elif isinstance(sub_values, dict):
                    for sub_sub_key, sub_sub_values in sub_values.items():
                        current_path = [main_key, sub_key, sub_sub_key]
                        if isinstance(sub_sub_values, data_type):
                            paths_to_examine.append({
                                'path': current_path,
                                'values': sub_sub_values
                            })
    return paths_to_examine

def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

def get_abbr(s):
    abbr = s[0]
    prev_char = s[0]
    for c in s[1:]:
        if prev_char in ['_', ' ']:
            abbr += c
        prev_char = c
    return abbr
