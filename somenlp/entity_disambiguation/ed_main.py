import csv
import re
import nltk
import json
import pandas as pd

from itertools import combinations

from somenlp.entity_disambiguation import EntityDisambiguationFeatureGenerator
from somenlp.entity_disambiguation.clustering import Clustering

def main(data, gold_standard, dbpedia):
    """Perform entity disambigation

    Args:
        data (Posix path): path to input data
        gold_standard (Posix path): path to labeled gold standard 
        dbpedia (Posix path): path to DBpedia distant supervision information
    """
    print("Loading input..")
    with data.open() as j_in:
        mention_list = json.load(j_in)
        for idx, ent in enumerate(mention_list): 
            ent['linking_id'] = idx

    print("Start calculating features")
    entity_disambiguator = EntityDisambiguationFeatureGenerator(dbpedia)
    feature_matrices = entity_disambiguator.apply_features(mention_list)

    print("Start clustering")
    clustering = Clustering(mention_list, feature_matrices)
    clustering.perform_clustering()

    cluster_ids = []
    for cluster in clustering.clusters:
        cluster_ids.append([])
        for mention in cluster:
            cluster_ids[-1].append(mention['linking_id'])

    print("Comparing to gold annotation")
    with gold_standard.open() as gold_data_f:
        gold_data = json.load(gold_data_f)
    for mention in mention_list:
        for entry in gold_data: 
            if mention['mention'] == entry['mention'] and mention['sentence_id'] == entry['sentence_id'] and mention['beg'] == entry['beg']:
                mention['link'] = entry['link']

    tp, fp, fn, tn = 0, 0, 0, 0
    tp_same_name, fp_same_name, fn_same_name, tn_same_name = 0, 0, 0, 0
    tp_diff_name, fp_diff_name, fn_diff_name, tn_diff_name = 0, 0, 0, 0
    mention_pairs = combinations(mention_list, 2)
    for pair in mention_pairs:
        id_one = pair[0]['linking_id']
        id_two = pair[1]['linking_id']
        link_true = pair[0]['link'] == pair[1]['link']
        link_pred = False
        matched_in_cluster_count = 0
        for cluster in cluster_ids:
            if id_one in cluster and id_two in cluster:
                matched_in_cluster_count += 1
                link_pred = True
        if link_pred and matched_in_cluster_count > 1:
            print("This should not happen")

        if link_true and link_pred:
            tp += 1
        elif link_true and not link_pred:
            fn += 1
            print(pair)
        elif not link_true and link_pred:
            fp += 1
        else:
            tn += 1

        if pair[0]['mention'] == pair[1]['mention']:
            if link_true and link_pred:
                tp_same_name += 1
            elif link_true and not link_pred:
                fn_same_name += 1
            elif not link_true and link_pred:
                fp_same_name += 1
            else:
                tn_same_name += 1
        else:
            if link_true and link_pred:
                tp_diff_name += 1
            elif link_true and not link_pred:
                fn_diff_name += 1
            elif not link_true and link_pred:
                fp_diff_name += 1
            else:
                tn_diff_name += 1

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