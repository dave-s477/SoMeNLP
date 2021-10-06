from somenlp.entity_disambiguation.efficient_prediction import BLOCK_SIZE
import torch
import torch.nn as nn
import torch.optim as optim
torch.multiprocessing.set_sharing_strategy('file_system')
import json
import math
import pickle
import os
from somenlp.entity_disambiguation.model import DisambiguationModel
import sys
import random
import itertools
import time as timeit
import numpy as np

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, IterableDataset, Dataset
from itertools import combinations
from articlenizer.util import chunk_list

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.metrics import classification_report
from somenlp.entity_disambiguation import LinkingData, EntityDisambiguationFeatureGenerator, FeatureWriter, ModelWrapper, Clustering, IntervalClustering, SimpleCluster
from somenlp.entity_disambiguation.efficient_prediction import IterDataset, worker_init_fn

def concat_features(vectors, labels=False):
    dim = -1
    if not vectors:
        raise(RuntimeError("No features were given"))
    feature_matrix = {}
    for k, v in vectors.items():
        if k not in feature_matrix:
            feature_matrix[k] = {} 
        for k2, v2 in v.items():
            if k2 not in feature_matrix[k]:
                feature_matrix[k][k2] = {}
            for idx, (k3, v3) in enumerate(v2.items()):
                if idx == 0:
                    matrix = np.expand_dims(v3, -1)
                elif labels and k3 == 'labels':
                    feature_matrix[k][k2]['labels'] = v3
                else:
                    matrix = np.append(matrix, np.expand_dims(v3, -1), axis=1) 
            if dim < 0:
                dim = matrix.shape[1]
            feature_matrix[k][k2]['features'] = matrix
    return feature_matrix, dim

def evaluate(entities, indicies, labels):
    pairs_to_compare = list(combinations(entities, 2))
    tp_same_name, fp_same_name, fn_same_name, tn_same_name = 0, 0, 0, 0
    tp_diff_name, fp_diff_name, fn_diff_name, tn_diff_name = 0, 0, 0, 0
    for pair, label in zip(pairs_to_compare, labels):
        exact_match = False
        if pair[0]['mention'] == pair[1]['mention']:
            exact_match = True
        pos = False
        for ind in indicies:
            if pair[0]['init_id'] in ind and pair[1]['init_id'] in ind:
                pos = True
        if pos and label:
            if exact_match:
                tp_same_name += 1
            else: 
                tp_diff_name += 1
        elif pos and not label:
            if exact_match:
                fp_same_name += 1
            else: 
                fp_diff_name += 1
        elif not pos and label:
            if exact_match:
                fn_same_name += 1
            else: 
                fn_diff_name += 1
        elif not pos and not label:
            if exact_match:
                tn_same_name += 1
            else: 
                tn_diff_name += 1
        tp = tp_same_name + tp_diff_name
        fp = fp_same_name + fp_diff_name
        fn = fn_same_name + fn_diff_name
        tn = tn_same_name + tn_diff_name

    print('Overall: tp ', tp, ', fp ', fp, ', fn ', fn, ', tn ', tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = (2 * precision * recall) / (precision + recall)
    print('Precision: {}, Recall {}, FScore: {}\n'.format(
        round(precision, 2), 
        round(recall, 2),
        round(fscore, 2)
    ))

    print('Same name: tp ', tp_same_name, ', fp ', fp_same_name, ', fn ', fn_same_name, ', tn ', tn_same_name)
    precision = tp_same_name / (tp_same_name + fp_same_name)
    recall = tp_same_name / (tp_same_name + fn_same_name)
    fscore = (2 * precision * recall) / (precision + recall)
    print('Precision: {}, Recall {}, FScore: {}\n'.format(
        round(precision, 2), 
        round(recall, 2),
        round(fscore, 2)
    ))

    print('Diff name: tp ', tp_diff_name, ', fp ', fp_diff_name, ', fn ', fn_diff_name, ', tn ', tn_diff_name)
    precision = tp_diff_name / (tp_diff_name + fp_diff_name)
    recall = tp_diff_name / (tp_diff_name + fn_diff_name)
    fscore = (2 * precision * recall) / (precision + recall)
    print('Precision: {}, Recall {}, FScore: {}\n'.format(
        round(precision, 2), 
        round(recall, 2),
        round(fscore, 2)
    ))

def generator_fct(sample_num, start_row, end_row):
    for idx_i in range(start_row, end_row):
            for idx_j in range(idx_i+1, sample_num):
                yield idx_i, idx_j       

def main(config, curr_time, ncores, gpu):
    if gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(gpu))
    if config['gold']:
        read_labels = True
    else:
        read_labels = False
    save_path = '{}/{}/{}'.format(
        config['logdir'].rstrip('/'),
        config['name'].rstrip('/'),
        curr_time
    )
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open('{}/config.json'.format(save_path), 'w') as j_out:
        json.dump(config, j_out, indent=4)
    linking_data = LinkingData(config)
    
    feature_generator = EntityDisambiguationFeatureGenerator(config['dbpedia'])
    input_dim = len(feature_generator.string_features_to_extract) + len(feature_generator.context_features_to_extract)
    print("Augmentated Dataset has {} samples".format(len(linking_data.all)))
    test_set_size = int(len(linking_data.all) * 0.18)
    
    model_w = ModelWrapper(config['model'], input_dim, save_path, device)
    print("MODEL")
    print(model_w.model)
    print()
    if config['model']['checkpoint']:
        print("Loading model for disambiguation.. (or re-training)")
        model_w.load()

    for e in range(config['model']['epochs']):
        model_w.epoch += 1
        print("At Epoch {} ({})".format(e, model_w.epoch))
        print(test_set_size)
        train_generator = generator_fct(len(linking_data.all), test_set_size, len(linking_data.all))
        count = 0
        while True:
            print("At block {}".format(count))
            count += 1
            block = list(itertools.islice(train_generator, BLOCK_SIZE))
            if not block:
                break
            train_dataset = IterDataset(list(block), linking_data.all, feature_generator, None, labels=linking_data.link_lookup_table)
            train_loader = DataLoader(train_dataset, batch_size=config['model']['batch_size'], num_workers=ncores, pin_memory=True)
            model_w.train(train_loader)
        
        print("Starting evaluation..")
        test_generator = generator_fct(len(linking_data.all), 0, test_set_size)
        count = 0
        true = []
        predictions = []
        while True:
            print("At block {}".format(count))
            count += 1
            block = list(itertools.islice(test_generator, BLOCK_SIZE))
            if not block:
                break
            test_dataset = IterDataset(list(block), linking_data.all, feature_generator, None, labels=linking_data.link_lookup_table)
            test_loader = DataLoader(test_dataset, batch_size=config['model']['batch_size'], num_workers=ncores, pin_memory=True)
            t, p = model_w.test(test_loader)
            true.extend(t)
            predictions.extend(p)
        model_w.eval(true, predictions, epoch=e)

    print("Starting evaluation..")
    test_generator = generator_fct(len(linking_data.all), 0, test_set_size)
    count = 0
    true = []
    predictions = []
    while True:
        print("At block {}".format(count))
        count += 1
        block = list(itertools.islice(test_generator, BLOCK_SIZE))
        if not block:
            break
        test_dataset = IterDataset(list(block), linking_data.all, feature_generator, None, labels=linking_data.link_lookup_table)
        test_loader = DataLoader(test_dataset, batch_size=config['model']['batch_size'])
        t, p = model_w.test(test_loader)
        true.extend(t)
        predictions.extend(p)
    model_w.eval(true, predictions, epoch=0, write=False)
    model_w.save()
