from itertools import combinations

class Clustering:
    """Cluster a set of entites
    """
    def __init__(self, entities, features):
        """Init

        Args:
            entities (list): list of entity mentions with additional information
            features (dictionary): dictionary of triangular numpy matricies containing features for clustering 
        """
        self.entities = entities
        self.features = features
        self.clusters = [[x] for x in entities]

    def match_clusters(self, clusters_to_match):
        """Combine internal clusters based on list of matching clusters

        Args:
            clusters_to_match (list): tuples of clusters that have been matched
        """
        l = clusters_to_match
        newl = []

        for ll in l:
            match = False
            for nl in newl:
                if bool(set(nl) & set(ll)):
                    nl.extend(ll)
                    match = True
                    break
            if not match:
                newl.append(ll)
        cluster_groups = [set(l) for l in newl]

        new_clusters = []
        matched_clusters = [False for x in self.clusters]
        for cluster_group in cluster_groups:
            cluster = []
            for elem in cluster_group:
                cluster.extend(self.clusters[elem])
                matched_clusters[elem] = True
            new_clusters.append(cluster)
        for m, c in zip(matched_clusters, self.clusters):
            if not m:
                new_clusters.append(c)
        self.clusters = new_clusters

    def perform_clustering(self):
        """Perform clustering based on fixed manually established rules
        """
        clusters_to_match = []
        matched = False
        for idx_1, c_1 in enumerate(self.clusters): 
            for idx_2, c_2 in enumerate(self.clusters[idx_1+1:]):
                matched = False
                for c_1_element in c_1:
                    if matched:
                        break
                    for c_2_element in c_2:
                        if matched:
                            break
                        if self.features['Levenshtein'][c_1_element['linking_id'], c_2_element['linking_id']] == 0 and self.features['URL_Substring_pos'][c_1_element['linking_id'], c_2_element['linking_id']] and self.features['Developer_Substring_pos'][c_1_element['linking_id'], c_2_element['linking_id']]:
                            clusters_to_match.append([idx_1, idx_2+idx_1+1])
                            matched = True
        self.match_clusters(clusters_to_match)

        clusters_to_match = []
        matched = False
        for idx_1, c_1 in enumerate(self.clusters): 
            for idx_2, c_2 in enumerate(self.clusters[idx_1+1:]):
                matched = False
                for c_1_element in c_1:
                    if matched:
                        break
                    for c_2_element in c_2:
                        if matched:
                            break
                        if self.features['URL_Levenshtein'][c_1_element['linking_id'], c_2_element['linking_id']] == 0 or ( self.features['Developer_Levenshtein'][c_1_element['linking_id'], c_2_element['linking_id']] == 0 and ( self.features['Substring'][c_1_element['linking_id'], c_2_element['linking_id']] or self.features['Levenshtein'][c_1_element['linking_id'], c_2_element['linking_id']] < 3)):
                            clusters_to_match.append([idx_1, idx_2+idx_1+1])
                            matched = True
        self.match_clusters(clusters_to_match)

        clusters_to_match = []
        matched = False
        for idx_1, c_1 in enumerate(self.clusters): 
            for idx_2, c_2 in enumerate(self.clusters[idx_1+1:]):
                matched = False
                for c_1_element in c_1:
                    if matched:
                        break
                    for c_2_element in c_2:
                        if matched:
                            break
                        if self.features['DBpedia_altname'][c_1_element['linking_id'], c_2_element['linking_id']]:
                            clusters_to_match.append([idx_1, idx_2+idx_1+1])
                            matched = True
        self.match_clusters(clusters_to_match)

        clusters_to_match = []
        matched = False
        for idx_1, c_1 in enumerate(self.clusters): 
            for idx_2, c_2 in enumerate(self.clusters[idx_1+1:]):
                matched = False
                for c_1_element in c_1:
                    if matched:
                        break
                    for c_2_element in c_2:
                        if matched:
                            break
                        if self.features['Abbreviation'][c_1_element['linking_id'], c_2_element['linking_id']] or ( self.features['Norm_string_Levenshtein'][c_1_element['linking_id'], c_2_element['linking_id']] == 0 and self.features['URL_Substring_pos'][c_1_element['linking_id'], c_2_element['linking_id']] and self.features['Developer_Substring_pos'][c_1_element['linking_id'], c_2_element['linking_id']] ):
                            clusters_to_match.append([idx_1, idx_2+idx_1+1])
                            matched = True
        self.match_clusters(clusters_to_match)
