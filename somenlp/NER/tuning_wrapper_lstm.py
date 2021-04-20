import pickle
import random 
import argparse
import json

from os.path import exists, join
from shutil import copytree

from itertools import product

import main_lstm as lstm
import main_lstm_with_features as feat_lstm
import main_lstm_combined as comb_lstm

def generate_all_configs(tuning_config):
    model_confs = [dict(zip(tuning_config["model"], v)) for v in product(*tuning_config["model"].values())]
    all_configs = []
    for mc in model_confs:
        mc['test_name'] = get_model_name(mc)
        all_configs.append({
            "mode": tuning_config['mode'],
            "data": tuning_config['data'],
            "model": mc,
            "analyses": tuning_config['analyses']
        })
    return all_configs

def get_abbr(s):
    abbr = s[0]
    prev_char = s[0]
    for c in s[1:]:
        if prev_char == '_':
            abbr += c
        prev_char = c
    return abbr

def get_model_name(model_config):
    base_name = 'O'
    for k, v in model_config.items():
        corr_v = v
        if isinstance(corr_v, str):
            if not corr_v:
                corr_v = 'none'
            else:
                corr_v = corr_v.replace(r'/', r'-')
                corr_v = corr_v.replace(r'.', r'-')
                corr_v = corr_v.replace(r'>', r'')
                corr_v = corr_v.replace(r'<', r'')
            
        if isinstance(corr_v, dict):
            corr_s = ''
            for k2, v2 in corr_v.items():
                corr_v2 = v2
                if isinstance(corr_v2, str):
                    corr_v2 = corr_v2.replace(r'/', r'-')
                corr_s = '{}_{}_{}'.format(corr_s, get_abbr(k2), corr_v2)
            corr_v = corr_s

        if isinstance(corr_v, bool):
            corr_v = str(corr_v)[0]
        base_name = '{}_{}_{}'.format(base_name, get_abbr(k), corr_v)
    if len(base_name) > 255:
        print(RuntimeWarning("Maximum naming lengths is reached, consider chaning the setup."))
    return base_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Run hyper-parameter tuning for Bi-LSTM-CRF.")

    parser.add_argument("--log-loc", default='', help="Name of log-dir.")
    parser.add_argument("--reset", action='store_true', help="Use new configs.")
    parser.add_argument("--config", required=True, help="name of config")

    args = parser.parse_args()

    with open(args.config, 'r') as t_file:
        config = json.load(t_file) 

    for key, value in config['data'].items():
        if key in ['pretrain', 'train', 'devel', 'test']:
            if value['text'] and not value['text'].startswith('/'):
                value['text'] = config['data']['base_dir'] + value['text'] if value['text'] else ''
                value['features'] = config['data']['base_dir'] + value['features'] if value['features'] else ''
                value['labels'] = config['data']['base_dir'] + value['labels'] if value['labels'] else ''
                if 'prepro' in value.keys() and 'arg' in value['prepro'].keys():
                    value['prepro']['arg'] = config['data']['base_dir'] + value['prepro']['arg']
                    value['relations']['tag_names'] = config['data']['base_dir'] + value['relations']['tag_names']
                    value['relations']['relations'] = config['data']['base_dir'] + value['relations']['relations']

    training_confs = generate_all_configs(config)

    for c in training_confs:
        print("Training a model with config:")
        print(json.dumps(c, indent=4))
        if 'combined' in args.config:
            print("Running a combined LSTM")
            comb_lstm.train(c)
        elif 'with_features' in args.config:
            print("Running a custom feature LSTM")
            feat_lstm.train(c)
        else:
            print("Running a plain LSTM")
            lstm.train(c)