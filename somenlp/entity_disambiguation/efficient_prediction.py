import torch
import torch.nn as nn
import os
import json
import pickle
import random
import copy
import time
import re
import math
import numpy as np
import nltk
import itertools
#nltk.download('stopwords')

from functools import partial
from nltk.corpus import stopwords
from torch.utils.data import DataLoader, IterableDataset, Dataset
#torch.multiprocessing.set_sharing_strategy('file_system')
from heapq import merge
from pathlib import Path

from . import EntityDisambiguationFeatureGenerator
from .model import DisambiguationModel
from sklearn.metrics import classification_report

BLOCK_SIZE = 5000000
STOPS = stopwords.words('english')

def normalize(s):
    norm_s = re.sub('[^0-9a-zA-Z]+', ' ', s.casefold()).rstrip('0123456789 ,.').lstrip(' ')
    norm_s = ' '.join([w for w in norm_s.split() if w not in STOPS])
    if not norm_s:
        norm_s = s
    return norm_s

def remove_spaces(s):
    replace_regex = re.compile(r'\s(?P<to_keep>[\+\-#™_/\d]+)\s?')
    matches = replace_regex.findall(s)
    return replace_regex.sub(r'\g<to_keep>', s)

class ReducedSampleSet():
    def __init__(self, in_paths, overview_file, save_path='/tmp'):
        self.output_path = '{}/reduced_features.json'.format(save_path)
        if os.path.isfile(self.output_path):
            self._load()
        else:
            self._sample_overview(overview_file)
            self._generate(in_paths)

    def _sample_overview(self, overview_file):
        self.sample_overview = {}
        with open(overview_file, 'r') as f_in:
            for line in f_in:
                num, key = line.rstrip().split(maxsplit=1)
                self.sample_overview[key] = int(num)

    def _load(self):
        with open(self.output_path, 'r') as j_in:
            self.sample_set = json.load(j_in)

    def _save(self):
        with open(self.output_path, 'w') as j_out:
            json.dump(self.sample_set, j_out, indent=4)

    def _generate(self, in_paths):
        file_count = 0
        self.sample_set = {}
        print("Gathering files..")
        for p in in_paths:
            file_list = Path(p).rglob('*.linking')
            for f in file_list:
                file_count += 1
                if file_count % 10000 == 0:
                    print("At file {}".format(file_count))
                with f.open() as f_in:
                    entities = json.load(f_in)
                for entity in entities:
                    name = entity.pop("mention")
                    if name not in self.sample_set:
                        self.sample_set[name] = {
                            "mention": name,
                            "string": remove_spaces(name),
                            "norm": normalize(name),
                            "contexts": []
                        }
                    if len(self.sample_set[name]['contexts']) < 5:
                        self.sample_set[name]['contexts'].append(copy.deepcopy(entity))
                    else:
                        keep_val = random.random()
                        occurrence_num = self.sample_overview[name] if name in self.sample_overview else 1000
                        if keep_val <= 1 / occurrence_num:
                            #print("Replace a value of {} with {} and num {}".format(name, keep_val, self.sample_overview[name]))
                            keep_pos = random.randint(0, 4)
                            self.sample_set[name]['contexts'][keep_pos] = copy.deepcopy(entity)
        self._save()

# class IterDataset(IterableDataset):
#     def __init__(self, data, sample_set, feature_calc, compare_set, labels=None):
#         self.data = data
#         self.sample_set = sample_set
#         self.feature_calc = feature_calc
#         self.compare_set = compare_set
#         self.labels = labels

#     def __iter__(self):
#         for x in self.data:
#             if self.compare_set is None:
#                 sample = [self.sample_set[x[0]], self.sample_set[x[1]]]
#             else:
#                 sample = [self.sample_set[x[0]], self.compare_set[x[1]]]
#             features = self.feature_calc.features_for_pair(sample)
#             origin = [sample[0]['origin_id'], sample[1]['origin_id']]
#             if self.labels is None:
#                 label = 0
#             else:
#                 label = self.labels[sample[0]['origin_id']][sample[1]['origin_id']]
#             yield torch.tensor(features), label, x, origin 

class IterDataset(Dataset):
    def __init__(self, data, sample_set, feature_calc, compare_set, labels=None):
        self.data = data
        self.sample_set = sample_set
        self.feature_calc = feature_calc
        self.compare_set = compare_set
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.compare_set is None:
            sample = [self.sample_set[x[0]], self.sample_set[x[1]]]
        else:
            sample = [self.sample_set[x[0]], self.compare_set[x[1]]]
        features = self.feature_calc.features_for_pair(sample)
        origin = [sample[0]['origin_id'], sample[1]['origin_id']]
        if self.labels is None:
            label = 0
        else:
            label = self.labels[sample[0]['origin_id']][sample[1]['origin_id']]
        return torch.tensor(features), label, x, origin 

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = math.ceil(len(dataset.data) / worker_info.num_workers)
    dataset.data = dataset.data[worker_id * split_size:(worker_id + 1) * split_size]

class DistanceMap():
    def __init__(self, sample_set, dbpedia, model_config, start_row, end_row, threshold=0.01, compare_set=None, n_cores=8, save_path='/tmp', batch_size=500, device='cpu'):
        self.device = torch.device(device)
        self.distance_map = []
        self.sample_set = sample_set 
        self.compare_set = compare_set
        self.sample_num = len(self.sample_set)
        self.max_idx = int(self.sample_num * ( self.sample_num - 1 ) / 2)
        self.threshold = threshold
        self.cores = n_cores
        self.batch_size = batch_size
        self.start_row = start_row
        self.end_row = end_row
        self.dbpedia = dbpedia
        self.model_config = model_config
        self.output_path = '{}/distance_map_{}_{}.p'.format(save_path, start_row, end_row)

    def calculate_distance_map(self, compare=False):
        if os.path.isfile(self.output_path):
            self._load()
        else:
            self.feature_generator = EntityDisambiguationFeatureGenerator(self.dbpedia)
            self._setup_model(self.model_config)
            if not compare:
                self._generate()
            else:
                self._compare()
    
    def _setup_model(self, config):
        self.model = DisambiguationModel(len(self.feature_generator.string_features_to_extract) + len(self.feature_generator.context_features_to_extract), self.model_config['layer_sizes'], self.model_config['drop_outs'])
        if not 'checkpoint' in config or not config['checkpoint']:
            raise(RuntimeError("A pretrained model is required for prediction.."))
        checkpoint_data = torch.load(config['checkpoint'], map_location=self.device)
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.model.eval()

    def _load(self):
        with open(self.output_path, 'rb') as p_in:
            self.distance_map = pickle.load(p_in)

    def _save(self):
        with open(self.output_path, 'wb') as p_out:
            pickle.dump(self.distance_map, p_out)

    def _generator(self):
        for idx_i in range(self.start_row, self.end_row):
            for idx_j in range(idx_i+1, len(self.sample_set)):
                yield idx_i, idx_j

    def _compare_generator(self):
        for idx_i in range(len(self.sample_set)):
            for idx_j in range(len(self.compare_set)):
                yield idx_i, idx_j

    def _prune_predictions(self, predictions):
        pruned = []
        last_source = []
        for prediction in predictions:
            if prediction[2] != last_source:
                pruned.append(prediction)
            else:
                if prediction[0] < pruned[-1][0]:
                    pruned[-1][0] = prediction[0]
            last_source = prediction[2]
        return pruned

    def _generate(self):
        generator = self._generator()
        self._run_generation(generator)
    
    def _compare(self):
        print("Doing compare")
        generator = self._compare_generator()
        self._run_generation(generator)

    def _run_generation(self, generator):
        count = 0
        while True:
            print("At block {}".format(count))
            count += 1
            start = time.time()
            block = list(itertools.islice(generator, BLOCK_SIZE))
            if not block:
                break
            iterable_dataset = IterDataset(list(block), self.sample_set, self.feature_generator, self.compare_set)
            loader = DataLoader(iterable_dataset, batch_size=self.batch_size, num_workers=self.cores, pin_memory=True)#self.batch_size, self.cores)
            predictions = []
            with torch.no_grad():
                for sample in loader:
                    #print(type(sample))
                    pred = self.model(sample[0].to(self.device))
                    pred_class = 1 - torch.squeeze(torch.sigmoid(pred)).cpu().numpy()
                    results = [[x, [int(idx_x), int(idx_y)], [int(org_x), int(org_y)]] for x, idx_x, idx_y, org_x, org_y in zip(pred_class, sample[2][0], sample[2][1], sample[3][0], sample[3][1]) if x <= self.threshold]
                    predictions.extend(results)
                    #print(results)
            print("preds")
            print(len(predictions))
            # pruned_predictions = self._prune_predictions(predictions)
            # print("pruned")
            # print(len(pruned_predictions))
            predictions.sort()
            self.distance_map = list(merge(self.distance_map, predictions))
            end = time.time()
            print("Took {}".format(round(end-start, 5)))
                
        self._save()

class EfficientClustering():
    def __init__(self, sorted_list, threshold, sample_set, save_path='/tmp', ncores=8):
        self.dim = len(sample_set)
        self.entities = sample_set
        self.threshold = threshold
        self.output_path = '{}/clusters.json'.format(save_path)
        self.clusters = list(range(len(sample_set)))
        self.features = sorted_list  
        self.ncores = ncores
        self.reverse_index = {}     
    
    def cluster(self):
        if os.path.isfile(self.output_path):
            self._load()
        else:
            self._cluster()
    
    def _load(self):
        with open(self.output_path, 'r') as j_in:
            self.clusters = json.load(j_in)

    def _save(self):
        with open(self.output_path, 'w') as j_out:
            json.dump(self.clusters, j_out, indent=4)
    
    def _get_cluster_idx(self, idx):
        cluster_value = self.clusters[idx]
        while idx != cluster_value:
            idx = cluster_value
            cluster_value = self.clusters[idx] 
        return idx

    def _match_clusters(self, matches):
        clusters_to_match = [set(x) for x in matches]
        global_index = len(clusters_to_match) - 1
        while global_index > 0:
            next_match = clusters_to_match[global_index]
            for idx, match in enumerate(clusters_to_match[:global_index]):
                if next_match & match:
                    new_matches = next_match.union(match)
                    clusters_to_match[idx] = new_matches
                    clusters_to_match.pop(global_index)
                    break
            global_index -= 1
        return clusters_to_match

    def _generator(self, start, end, clusters):
        for idx_i in range(start, end):
            for idx_j in range(idx_i+1, len(clusters)):
                yield idx_i, idx_j

    def _get_postprocessing_merges(self, rows, clusters):
        print("Len clusters {}".format(len(clusters)))
        gen = self._generator(rows[0], rows[1], clusters)
        merges = []
        for i in gen:
            if clusters[i[0]] & clusters[i[1]]:
                merges.append(i)
        return merges

    def _iterate_distances(self, threshold, cut_off_idx=0):
        start = time.time()
        if cut_off_idx > 0:
            features = self.features[cut_off_idx:]
        else:
            features = self.features
        for idx, i in enumerate(features):
            if (idx+1) % 10000 == 0:
                end = time.time()
                print("At {}: {}, last step took {}".format(idx, i, round(end-start, 4))) 
                start = end
            dist_val, pair_indices, _ = i
            if dist_val > threshold:
                break
            x_idx, y_idx = pair_indices
            x_pointer = self.clusters[x_idx]
            y_pointer = self.clusters[y_idx]
            if x_pointer == y_pointer:
                # already a correct link -> do nothing
                pass
            elif x_pointer == x_idx and y_pointer == y_idx:
                # two "original" samples -> new cluster
                self.clusters[x_idx] = len(self.clusters)
                self.clusters[y_idx] = len(self.clusters)
                self.reverse_index[len(self.clusters)] = [x_idx, y_idx]
                self.clusters.append(len(self.clusters))
            elif x_pointer != x_idx and y_pointer == y_idx:
                # first point does already belong to a cluster, second is "original" -> add second point to cluster
                self.clusters[y_idx] = x_pointer
                self.reverse_index[x_pointer].append(y_idx)
            elif y_pointer != y_idx and x_pointer == x_idx:
                # second point does already belong to a cluster, first is "original" -> add first point to cluster
                self.clusters[x_idx] = y_pointer
                self.reverse_index[y_pointer].append(x_idx)
            elif y_pointer != y_idx and x_pointer != x_idx:
                # both points already belong to clusters -> merge clusters (we will add one "empty" cluster in the result)
                reverse_y_pointers = self.reverse_index.pop(y_pointer)
                for p in reverse_y_pointers:
                    self.clusters[p] = x_pointer
                    self.reverse_index[x_pointer].append(p)
        return idx + cut_off_idx

    def _values_to_clusters(self):
        cluster_buckets = {}
        for i in range(len(self.clusters)-1, -1, -1):
            if i == self.clusters[i]:
                cluster_buckets[i] = {
                    'name': i,
                    'indices': [i],
                    'entities': []
                }
            else:
                bucket_idx = self._get_cluster_idx(i)
                cluster_buckets[bucket_idx]['indices'].append(i)
        return cluster_buckets

    def _print_values(self, n=100):
        for i in [x for x in self.features if x[0] < self.threshold][-100:]:
            dist, pair_indices, _ = i
            x_idx, y_idx = pair_indices
            print(dist)
            print('x = {}'.format(self.entities[x_idx]))
            print('y = {}'.format(self.entities[y_idx]))
            print()

    def _cluster(self):
        self._pre_cluster()

        print("Iterating distances..")
        self._iterate_distances(self.threshold)

        print("Writing values to clusters..")
        cluster_buckets = self._values_to_clusters()

        for bucket in cluster_buckets:
            cluster_buckets[bucket]['indices'] = [x for x in cluster_buckets[bucket]['indices'] if x < self.dim]

        for cluster in cluster_buckets:
            for idx in cluster_buckets[cluster]['indices']:
                cluster_buckets[cluster]['entities'].append(self.entities[idx])
        self.clusters = cluster_buckets

        print("Removing empty clusters..")
        self.clusters = {x:y for x,y in self.clusters.items() if y['entities']}

        for cluster, values in self.clusters.items():
            values['origin_ids'] = [x['origin_id'] for x in values['entities']]
        
        self._save()

    def _pre_cluster(self):
        print("Performing pre-clustering on exact string matches.")
        name_index = {}
        for idx, ent in enumerate(self.entities):
            if idx % 10000 == 0:
                print("At entity {}".format(idx))
            if ent['mention'] not in name_index:
                name_index[ent['mention']] = []
            name_index[ent['mention']].append(idx)
        for idx, (k, v) in enumerate(name_index.items()):
            if idx % 10000 == 0:
                print("At pre-cluster {}".format(idx))
            if len(v) > 1:
                cluster_id = len(self.clusters)
                self.reverse_index[cluster_id] = []
                self.clusters.append(cluster_id)
                for ent_id in v:
                    self.clusters[ent_id] = cluster_id
                    self.reverse_index[cluster_id].append(ent_id)
    
    def evaluate(self, gold_table, step_size, start_eval_threshold):
        print("Evaluating clustering..")
        self._pre_cluster()

        current_threshold = 0
        iteration_index = 0
        prev_num_clusters = len(self.clusters)
        while current_threshold < self.threshold:
            current_threshold = min(self.threshold, current_threshold + step_size)
            iteration_index = self._iterate_distances(current_threshold, cut_off_idx=iteration_index)
            if current_threshold >= start_eval_threshold and len(self.clusters) > prev_num_clusters:
                print("Added clusters - start evaluation")
                prev_num_clusters = len(self.clusters)
                eval_entities_with_clusters = [[x, y] for x, y in zip(self.entities, self.clusters) if 'eval' in x]
                print(len(eval_entities_with_clusters))
                # for x in eval_entities_with_clusters:
                #     print(x)
                #     print()
                # break
                preds = []
                trues = []
                for idx, entity in enumerate(eval_entities_with_clusters):
                    for comp_entity in eval_entities_with_clusters[idx+1:]:
                        pred_link = entity[1] == comp_entity[1]
                        preds.append(pred_link)
                        true_link = gold_table[entity[0]['origin_id']][comp_entity[0]['origin_id']]
                        trues.append(true_link)
                
                print("Threshold {}".format(current_threshold))
                # print("Iteration idx {}".format(iteration_index))
                print("Num clusters: {} (not correct - just for debugging)".format(len(self.clusters)))
                # print(classification_report(trues, preds))
                output_dict = classification_report(trues, preds, output_dict=True)
                logging_string = 'Graph_out:\t{},{},{},{}'.format(current_threshold, output_dict['True']['precision'], output_dict['True']['recall'], output_dict['True']['f1-score'])
                print(logging_string)
                print()

    # def _post_pro_merge(self):
        # total_size = len(self.clusters) * (len(self.clusters) - 1) / 2
        # max_samples = math.ceil(total_size / self.ncores)
        # row_overview = [0]
        # curr_size = 0
        # for row in range(len(self.clusters)-1):
        #     curr_size += len(self.clusters)-1-row
        #     if curr_size > max_samples:
        #         row_overview.append(row)
        #         curr_size = 0
        # row_overview.append(row)
        # row_pairs = [[row_overview[x-1], row_overview[x]] for x in range(1, len(row_overview))]
        # print(row_overview)
        # print("pairs")
        # print(row_pairs)

        # with Pool(len(row_overview)) as p:
        #     origin_ids = [x['origin_ids'] for x in self.clusters.values()]
        #     fct = partial(self._get_postprocessing_merges, clusters=origin_ids)
        #     post_processing_merges = p.map(fct, row_pairs)

        # TODO -> flatten list
        # post_processing_merges = [item for sublist in post_processing_merges for item in sublist]

        # post_processing_merges = []
        # for idx_x, (cluster_x, values_x) in enumerate(self.clusters.items()):
        #     if idx_x % 5000 == 0:
        #         print("At cluster {}".format(idx_x))
        #     for cluster_y, values_y in list(self.clusters.items())[idx_x+1:]:
        #         for ent_x in values_x['entities']:
        #             matched = False
        #             for ent_y in values_y['entities']:
        #                 if ent_x['mention'] == ent_y['mention']:
        #                     matched = True
        #        #if values_x['origin_ids'] & values_y['origin_ids']:
        #                     post_processing_merges.append([cluster_x, cluster_y])
        #                     break
        #             if matched:
        #                 break
            
        # p_merges = self._match_clusters(post_processing_merges)
        # for ppm in p_merges:
        #     ppm = list(ppm)
        #     for i in ppm[1:]:
        #         values = self.clusters.pop(i)
        #         self.clusters[ppm[0]]['indices'].extend(values['indices'])
        #         self.clusters[ppm[0]]['entities'].extend(values['entities'])
        #         self.clusters[ppm[0]]['origin_ids'].update(values['origin_ids'])
