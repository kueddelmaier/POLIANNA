import json
import os
from collections import namedtuple
import sys

sys.path.insert(0,'../..')

from src.experiment_utils.helper_classes import token, span, repository

def get_config(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)

    return config


def load_json(path):
    with open(path) as f:
        json_contents = json.load(f)
    return json_contents


def config_to_namedtuple(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = config_to_namedtuple(value)
        return namedtuple('GenericDict', obj.keys())(**obj)
    elif isinstance(obj, list):
        return [config_to_namedtuple(item) for item in obj]
    else:
        return obj


def remove_span_doublicates(span_list):

    hash_table = {}
    for span_ in span_list:
        hash_value = hash(span_)
        if hash_value in hash_table:
            del span_
        else:
            hash_table[hash_value] = span_
        
    ret = list(hash_table.values())

    if len(ret) < len(span_list):
        print('removed the following doublicates:')
        print([x for x in span_list if x not in ret])
    return ret

