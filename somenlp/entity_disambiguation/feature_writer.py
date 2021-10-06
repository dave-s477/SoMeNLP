import json
import os
import numpy as np

class FeatureWriter:
    def __init__(self, save_dir):
        self.output_path = save_dir

    def read_features(self, path):
        with open(path, 'r') as j_in:
            feature_locations = json.load(j_in)
        feature_matrices = {}
        for k,v in feature_locations.items():
            if k not in feature_matrices:
                feature_matrices[k] = {}
            for k2, v2 in v.items():
                if k2 not in feature_matrices[k]:
                    feature_matrices[k][k2] = {}
                for k3, v3 in v2.items():
                    feature_matrices[k][k2][k3] = np.load(v3)
        return feature_matrices

    def write_features(self, matrices):
        os.makedirs(self.output_path, exist_ok=True)
        save_locations = {}
        for set_k, set_v in matrices.items():
            if set_k not in save_locations:
                save_locations[set_k] = {}
            for set_k2, set_v2 in set_v.items():
                if set_k2 not in save_locations[set_k]:
                    save_locations[set_k][set_k2] = {}
                for feature_k, feature_m in set_v2.items():
                    output_name = '{}/{}_{}_{}'.format(self.output_path, set_k, set_k2, feature_k)
                    save_locations[set_k][set_k2][feature_k] = output_name + '.npy'
                    np.save(output_name, feature_m)
        with open(self.output_path + '/save_locations.json', 'w') as j_out:
            json.dump(save_locations, j_out, indent=4)

    def save_triangular_matrix(self, tri_matrix, save_location):
        """ This function saves a triangular matrix in a flat representation so unnecessary zeros can be excluded. Only necessary indicies from the matrix are extracted line by line. 
        
        Arguments:
            tri_matrix (np.matrix with an arbitrary d_type): the matrix to save
            save_location (string): path to output file
        """
        indices = np.triu_indices(tri_matrix.shape[0], 1)
        flat_representation = tri_matrix[indices]
        np.save(save_location, flat_representation)