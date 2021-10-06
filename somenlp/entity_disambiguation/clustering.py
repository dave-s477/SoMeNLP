import numpy as np
import math
import time
import bisect
import sys

from statistics import mean
from multiprocessing import Pool

class SimpleCluster:
    """Cluster a set of entites
    """
    def __init__(self, config, entities, features):
        """Init

        Args:
            entities (list): list of entity mentions with additional information
            features (dictionary): dictionary of triangular numpy matricies containing features for clustering 
        """
        self.config = config
        self.features = []
        for idx, i in enumerate(features):
            if i <= config['threshold']:
                self.features.append([i[0], idx])
        self.features.sort()
        self.dim = len(entities)
        self.entities = entities
        self.reverse_index = {}
        #self.clusters = {idx: {'entities': [x]} for idx, x in enumerate(entities)}
        self.clusters = list(range(len(entities)))
        #self.clusters = {x: x for x in range(len(entities))}

    def _get_eval_indices(self):
        eval_indices = []
        for _, cluster in self.clusters.items():
            indices = [x['init_id'] for x in cluster['entities']]
            eval_indices.append(indices)
        return eval_indices

    def _idx_to_matrix_pos(self, idx):
        i = self.dim - 2 - int(math.sqrt(-8*idx + 4*self.dim*(self.dim-1)-7)/2.0 - 0.5)
        j = idx + i + 1 - self.dim*(self.dim-1)/2 + (self.dim-i)*((self.dim-i)-1)/2
        return i, int(j)

    def _get_cluster_idx(self, idx):
        #print("idx..")
        #print(idx)
        cluster_value = self.clusters[idx]
        #print(cluster_value)
        while idx != cluster_value:
            idx = cluster_value
            #print(idx)
            cluster_value = self.clusters[idx] 
        #print()
        return idx

    def cluster(self):
        for i in self.features:
            _, pair_index = i 
            x_idx, y_idx = self._idx_to_matrix_pos(pair_index)
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
                #self.clusters = [i if i == idx or i != y_pointer else x_pointer for idx, i in enumerate(self.clusters)]
                #self.clusters[:self.dim] = [i if i == idx or i != y_pointer else x_pointer for idx, i in enumerate(self.clusters[:self.dim])]
                reverse_y_pointers = self.reverse_index.pop(y_pointer)
                for p in reverse_y_pointers:
                    self.clusters[p] = x_pointer
                    self.reverse_index[x_pointer].append(p)

        #print(len(self.clusters))
        cluster_buckets = {}
        for i in range(len(self.clusters)-1, -1, -1):
            #print(i)
            #print(self.clusters[i])
            if i == self.clusters[i]:
                cluster_buckets[i] = {
                    'name': i,
                    'indices': [i],
                    'entities': []
                }
            else:
                bucket_idx = self._get_cluster_idx(i)
                cluster_buckets[bucket_idx]['indices'].append(i)
        #print(len(cluster_buckets))
        for bucket in cluster_buckets:
            #print(bucket)
            #print(cluster_buckets[bucket])
            cluster_buckets[bucket]['indices'] = [x for x in cluster_buckets[bucket]['indices'] if x < self.dim]
            #print(cluster_buckets[bucket])
            #print()
        for cluster in cluster_buckets:
            for idx in cluster_buckets[cluster]['indices']:
                cluster_buckets[cluster]['entities'].append(self.entities[idx])
        self.clusters = cluster_buckets

class Clustering:
    """Cluster a set of entites
    """
    def __init__(self, config, entities, features):
        """Init

        Args:
            entities (list): list of entity mentions with additional information
            features (dictionary): dictionary of triangular numpy matricies containing features for clustering 
        """
        self.config = config
        self.features = features
        # self._calculate_distances()
        self.threshold = config['threshold']
        self.dim = len(entities)
        self.clusters = {idx: {'entities': [x]} for idx, x in enumerate(entities)}
        self.new_clusters = list(self.clusters.keys())
        self.cluster_count = len(self.clusters)

    # def _calculate_distances(self):
    #     if self.config['feat_to_dist'] == 'average':
    #         self.distances = np.mean(self.features, axis=-1)
    #         print(self.distances)
    #     elif self.config['feat_to_dist'] == 'sum':
    #         self.distances = np.sum(self.features, axis=-1)
    #     else:
    #         raise(RuntimeError("Got unknown feat to dist config: {}".format(self.config['feat_to_dist'])))
    #     self.min = self.distances.min()
    #     print(self.min)
    #     self.max = self.distances.max()
    #     print(self.max)
    #     self.threshold = self.min + (self.max - self.min) * self.config['threshold']
    #     print(self.threshold)

    def _get_eval_indices(self):
        eval_indices = []
        for _, cluster in self.clusters.items():
            indices = [x['init_id'] for x in cluster['entities']]
            eval_indices.append(indices)
        return eval_indices

    def _get_matrix_index(self, x, y):
        if x > y:
            x1 = y
            x2 = x
        else:
            x1 = x
            x2 = y
        return int((x1 * self.dim) - ((x1+1) * ((x1+1) + 1) / 2 ) + x2)  

    def _calculate_distance(self, cluster_id):
        distances = {}
        for c_k, cluster in self.clusters.items():
            if c_k == cluster_id:
                distances[cluster_id] = 1.0
            else:
                cluster_distances = []
                for c_ent in cluster['entities']:
                    for id_ent in self.clusters[cluster_id]['entities']:
                        cluster_distances.append(self.features[self._get_matrix_index(c_ent['init_id'], id_ent['init_id'])])
                if self.config['linkage'] == 'single':
                    distances[c_k] = float(min(cluster_distances))
                elif self.config['linkage'] == 'complete':
                    distances[c_k] = float(max(cluster_distances))
                elif self.config['linkage'] == 'average':
                    distances[c_k] = mean([float(x) for x in cluster_distances])
                else:
                    raise(RuntimeError("Received unkown type of linkage {}".format(self.config['linkage'])))  
        return distances

    def _calculate_distances_cluster_parallel(self):
        pass

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

    def cluster(self):
        # max_count = 0
        while len(self.new_clusters) > 0: # and max_count < 1000:
            # max_count += 1
            if len(self.new_clusters) > 1:
                # calculate distance to all other clusters parallel with defined linkage type
                for cluster in self.new_clusters:
                    dist = self._calculate_distance(cluster)
                    self.clusters[cluster]['distances'] = dist
            else:
                # calculate distance to all other clusters parallel with defined linkage type
                self.clusters[self.new_clusters[0]]['distances'] = self._calculate_distance(self.new_clusters[0])
            # find smallest distance between all existing clusters.. test if clusters do still exist!
            smallest_dist = 1.0
            matches = []
            for c_k, cluster in self.clusters.items():
                keys_to_pop = []
                for dist_k, dist_value in cluster['distances'].items():
                    # TODO: test if clusters do still exist
                    if dist_k in self.clusters:
                        if math.isclose(smallest_dist, dist_value):
                            to_append = sorted([c_k, dist_k])
                            if to_append not in matches:
                                matches.append(to_append)
                        elif dist_value < smallest_dist:
                            smallest_dist = dist_value
                            to_append = sorted([c_k, dist_k])
                            matches = [to_append]
                    else:
                        keys_to_pop.append(dist_k)
                for k_pop in keys_to_pop:
                    cluster['distances'].pop(k_pop)
            if smallest_dist > self.threshold:
                print("Stopped clustering due to threshold criterion: T {}, D {}".format(round(self.threshold, 4), round(smallest_dist, 4)))
                break
            matched_clusters = self._match_clusters(sorted(matches))  

            # pop clusters to be merged from cluster list
            self.new_clusters = []
            for cluster_group in matched_clusters:
                new_entities = []
                for cluster_idx in cluster_group:
                    clust_entities = self.clusters.pop(cluster_idx)
                    new_entities.extend(clust_entities['entities'])
                self.clusters[self.cluster_count] = {
                    'entities': new_entities
                }
                self.new_clusters.append(self.cluster_count)
                self.cluster_count += 1

class IntervalClustering:
    """Cluster a set of entites
    """
    def __init__(self, config, entities, features):
        """Init

        Args:
            entities (list): list of entity mentions with additional information
            features (dictionary): dictionary of triangular numpy matricies containing features for clustering 
        """
        self.config = config
        self.features = features
        self.threshold = config['threshold']
        self.n_intervals = config['intervals']
        self.n_cores = config['n_cores']
        self.drop_below = config['drop_below_threshold']
        self.dim = len(entities)
        self.clusters = {idx: {'entities': [x]} for idx, x in enumerate(entities)}
        self.new_clusters = list(self.clusters.keys())
        self.cluster_count = len(self.clusters)
        self._calculate_intervals()
        self.cluster_distances = []
        self.cluster_distances_keys = []

    def _calculate_intervals(self):
        self.intervals = np.linspace(0, self.threshold, num=self.n_intervals, endpoint=True)

    def _get_eval_indices(self):
        eval_indices = []
        for _, cluster in self.clusters.items():
            indices = [x['init_id'] for x in cluster['entities']]
            eval_indices.append(indices)
        return eval_indices

    def _get_matrix_index(self, x, y):
        if x > y:
            x1 = y
            x2 = x
        else:
            x1 = x
            x2 = y
        return int((x1 * self.dim) - ((x1+1) * ((x1+1) + 1) / 2 ) + x2)  

    def _calculate_distance(self, cluster_id):
        distances = {}
        for c_k, cluster in self.clusters.items():
            if c_k == cluster_id:
                distances[cluster_id] = 1.0
            else:
                cluster_distances = []
                for c_ent in cluster['entities']:
                    for id_ent in self.clusters[cluster_id]['entities']:
                        cluster_distances.append(self.features[self._get_matrix_index(c_ent['init_id'], id_ent['init_id'])])
                if self.config['linkage'] == 'single':
                    distances[c_k] = float(min(cluster_distances))
                elif self.config['linkage'] == 'complete':
                    distances[c_k] = float(max(cluster_distances))
                elif self.config['linkage'] == 'average':
                    distances[c_k] = mean([float(x) for x in cluster_distances])
                else:
                    raise(RuntimeError("Received unkown type of linkage {}".format(self.config['linkage'])))  
        return distances

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

    def _calculate_distance(self, distance_tuple):
        single_distances = []
        for first_entity in self.clusters[distance_tuple[0]]['entities']:
            for second_entity in self.clusters[distance_tuple[1]]['entities']:
                single_distances.append(self.features[self._get_matrix_index(first_entity['init_id'], second_entity['init_id'])])
        if self.config['linkage'] == 'single':
            distance = float(min(single_distances))
        elif self.config['linkage'] == 'complete':
            distance = float(max(single_distances))
        elif self.config['linkage'] == 'average':
            distance = mean([float(x) for x in single_distances])
        else:
            raise(RuntimeError("Received unkown type of linkage {}".format(self.config['linkage'])))
        if self.drop_below and distance > self.threshold:
            return None
        return distance

    def _calculate_distances_cluster_parallel(self, distance_tuples):
        with Pool(self.n_cores) as p:
            distances = p.map(self._calculate_distance, distance_tuples)
        return distances

    def _update_distance_list(self):
        distances_to_calculate = set()
        print(len(self.new_clusters))
        while self.new_clusters:
            cluster = self.new_clusters.pop()
            distances_to_calculate.update([tuple([x, cluster]) if x < cluster else tuple([cluster, x]) for x in self.clusters.keys() if x != cluster])
        print(len(distances_to_calculate))
        distances = self._calculate_distances_cluster_parallel(distances_to_calculate)
        distances_with_keys = [[x, y] for x, y in zip(distances, distances_to_calculate) if x is not None]
        print("Benchmark")
        start = time.time()
        sorted(distances_with_keys) 
        end = time.time()
        print("First sorting option: {}".format(round(end-start, 4)))
        start = time.time()
        qsort(distances_with_keys) 
        end = time.time()
        print("First sorting option: {}".format(round(end-start, 4)))
        if not self.cluster_distances:
            distance_tuples = sorted(distances_with_keys) 
            self.cluster_distances = [x[0] for x in distance_tuples]
            self.cluster_distances_keys = [x[1] for x in distance_tuples]
        else:
            for distance, keys in distances_with_keys:
                insertion_index = bisect.bisect(self.cluster_distances, distance)
                self.cluster_distances.insert(insertion_index, distance)
                self.cluster_distances_keys.insert(insertion_index, keys)
            # TODO: use bisect to get right insertion point for each element

    def _remove_from_distance_list(self, elements):
        indices = []
        for idx, cluster_keys in enumerate(self.cluster_distances_keys):
            if cluster_keys[0] in elements or cluster_keys[1] in elements:
                indices.append(idx)
        for idx in reversed(indices):
            self.cluster_distances_keys.pop(idx)
            self.cluster_distances.pop(idx)

    def cluster(self):
        for interval in self.intervals[1:]:
            start = time.time()
            self._update_distance_list()


            # for cluster in self.new_clusters:
            #     dist = self._calculate_distance(cluster)
            #     self.clusters[cluster]['distances'] = dist
            end = time.time()
            print("Calculating distances took {}".format(round(end-start, 4)))
            start = end
            # # find smallest distance between all existing clusters.. test if clusters do still exist!
            # matches = []
            bisection_point = bisect.bisect_right(self.cluster_distances, interval)
            print(bisection_point)

            clusters_to_merge = self.cluster_distances_keys[:bisection_point]
            matched_clusters = self._match_clusters(clusters_to_merge)
            end = time.time()
            print("Matching clusters took {}".format(round(end-start, 4)))
            start = end
            self.cluster_distances_keys = self.cluster_distances_keys[bisection_point:]
            self.cluster_distances = self.cluster_distances[bisection_point:]
            clusters_to_remove = set(sorted([item for sublist in matched_clusters for item in sublist]))
            self._remove_from_distance_list(clusters_to_remove)

            # for c_k, cluster in self.clusters.items():
            #     keys_to_pop = []
            #     for dist_k, dist_value in cluster['distances'].items():
            #         # TODO: test if clusters do still exist
            #         if dist_k in self.clusters:
            #             if dist_value <= interval:
            #                 to_append = sorted([c_k, dist_k])
            #                 if to_append not in matches:
            #                     matches.append(to_append)
            #         else:
            #             keys_to_pop.append(dist_k)
            #     for k_pop in keys_to_pop:
            #         cluster['distances'].pop(k_pop)
            # matched_clusters = self._match_clusters(sorted(matches))  
            end = time.time()
            print("Removing entries took {}".format(round(end-start, 4)))
            start = end

            # # pop clusters to be merged from cluster list
            self.new_clusters = []
            for cluster_group in matched_clusters:
                new_entities = []
                for cluster_idx in cluster_group:
                    clust_entities = self.clusters.pop(cluster_idx)
                    new_entities.extend(clust_entities['entities'])
                self.clusters[self.cluster_count] = {
                    'entities': new_entities
                }
                self.new_clusters.append(self.cluster_count)
                self.cluster_count += 1
            end = time.time()
            print("New clustering took {}\n".format(round(end-start, 4)))
            start = end
