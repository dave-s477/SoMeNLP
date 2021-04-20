import csv
import numpy as np
import pandas as pd
import time
import re
import copy
import nltk
nltk.download('stopwords')

from itertools import product
from nltk.corpus import stopwords

from os.path import join, exists
from Levenshtein import distance as levenshtein_distance

class EntityDisambiguationFeatureGenerator:
    """Calculates features for entity disambiguation
    """
    def __init__(self, dbpedia):
        """Init

        Args:
            dbpedia (str): location of a pandas datafram containing DBpedia information for software entities
        """
        self.stops = stopwords.words('english')
        self.dbpedia_names = pd.read_csv(dbpedia, compression='gzip')
        self.dbpedia_names.drop(self.dbpedia_names.columns.difference(['unique','label']), 1, inplace=True)
        self.dbpedia_names.dropna(inplace=True)
        self.dbpedia_unique_mapping = {}
        self.dbpedia_label_mapping = {}
        for index, row in self.dbpedia_names.iterrows():
            if row['unique'] not in self.dbpedia_unique_mapping:
                self.dbpedia_unique_mapping[row['unique']] = set()
            self.dbpedia_unique_mapping[row['unique']].update([row['label']])
            if row['label'] not in self.dbpedia_label_mapping:
                self.dbpedia_label_mapping[row['label']] = set()
            self.dbpedia_label_mapping[row['label']].update([row['unique']])

        # each feature is represented as a dictionary, where 'function' is supposed
        # to be an actual python exectuable function that calculates the distance between a pair of words.
        # The 'data_type' is important to limit the memory usage. The basic (unnormalized) Levenshtein distance
        # can nicely be represented as a 8-bit Integer (assuming we are not going to have words with a 
        # difference larger than 255)
        self.features_to_extract = [
            {
                'name': 'Levenshtein',
                'function': self._levenshtein,
                'data_type': np.int16,
                'init': 100
            },{
                'name': 'URL_Levenshtein',
                'function': self._url_levenshtein,
                'data_type': np.int16,
                'init': 200
            },{
                'name': 'Substring',
                'function': self._substring,
                'data_type': np.bool,
                'init': False
            },{
                'name': 'URL_Substring',
                'function': self._url_substring,
                'data_type': np.bool,
                'init': False
            },{
                'name': 'URL_Substring_pos',
                'function': self._url_substring_pos,
                'data_type': np.bool,
                'init': False
            },{
                'name': 'Developer_Levenshtein',
                'function': self._developer_levenshtein,
                'data_type': np.int16,
                'init': 100
            },{
                'name': 'Developer_Substring',
                'function': self._developer_substring,
                'data_type': np.bool,
                'init': False
            },{
                'name': 'Developer_Substring_pos',
                'function': self._developer_substring_pos,
                'data_type': np.bool,
                'init': False
            },{ 
                'name': 'Norm_string_Levenshtein',
                'function': self._norm_string_levenshtein,
                'data_type': np.int16,
                'init': 100
            },{
                'name': 'Abbreviation',
                'function': self._abbreviation,
                'data_type': np.bool,
                'init': False
            },{
                'name': 'DBpedia_altname',
                'function': self._dbpedia_alt_names,
                'data_type': np.bool,
                'init': False
            }
        ]

    def _levenshtein(self, x, y):
        """Levenshtein distance between entity mentions

        Args:
            x (list): entity defining list
            y (list): entity defining list

        Returns:
            int: Levenshtein edit distance
        """
        return levenshtein_distance(x['mention'], y['mention'])

    def _substring(self, x, y):
        """Test substring relation between entity mentions

        Args:
            x (list): entity defining list
            y (list): entity defining list

        Returns:
            bool: result
        """
        if len(x['mention']) < 2 or len(y['mention']) < 2:
            return False
        if x['mention'] in y['mention'] or y['mention'] in x['mention']:
            return True
        else:
            return False

    def _norm_string_levenshtein(self, x, y):
        """Levenshtein distance on normalized mentions

        Args:
            x (list): entity defining list
            y (list): entity defining list

        Returns:
            [type]: [description]
        """
        norm_x = re.sub('[^0-9a-zA-Z]+', ' ', x['mention'].casefold()).rstrip('0123456789 ,.').lstrip(' ')
        norm_x = ' '.join([w for w in norm_x.split() if w not in self.stops])
        norm_y = re.sub('[^0-9a-zA-Z]+', ' ', y['mention'].casefold()).rstrip('0123456789 ,.').lstrip(' ')
        norm_y = ' '.join([w for w in norm_y.split() if w not in self.stops])
        return levenshtein_distance(norm_x, norm_y)

    def _url_levenshtein(self, x, y):
        """Levenshtein distance between urls associated with entities

        Args:
            x (list): entity defining list
            y (list): entity defining list

        Returns:
            int: edit distance
        """
        x_url = []
        for rel in x['relations']: 
            if rel['type'] == 'url_of':
                x_url.append(rel['string'])
        if not x_url:
            return 200
        y_url = []
        for rel in y['relations']: 
            if rel['type'] == 'url_of':
                y_url.append(rel['string'])
        if not y_url:
            return 200
        min_dist = 200
        for u_x in x_url: 
            for u_y in y_url:
                dist = levenshtein_distance(u_x, u_y)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _url_substring(self, x, y):
        """URL substring relation between urls associated with entities

        Args:
            x (list): entity defining list
            y (list): entity defining list

        Returns:
            bool: result
        """
        x_url = []
        for rel in x['relations']: 
            if rel['type'] == 'url_of':
                x_url.append(rel['string'])
        if not x_url:
            return False
        y_url = []
        for rel in y['relations']: 
            if rel['type'] == 'url_of':
                y_url.append(rel['string'])
        if not y_url:
            return False
        for u_x in x_url: 
            for u_y in y_url:
                if u_x in u_y or u_y in u_x:
                    return True
        return False

    def _url_substring_pos(self, x, y):
        """URL substring relation between urls associated with entities.
        Empty URL count as positive matches.

        Args:
            x (list): entity defining list
            y (list): entity defining list

        Returns:
            bool: result
        """
        x_url = []
        for rel in x['relations']: 
            if rel['type'] == 'url_of':
                x_url.append(rel['string'])
        if not x_url:
            return True
        y_url = []
        for rel in y['relations']: 
            if rel['type'] == 'url_of':
                y_url.append(rel['string'])
        if not y_url:
            return True
        for u_x in x_url: 
            for u_y in y_url:
                if u_x in u_y or u_y in u_x:
                    return True
        return False

    def _developer_levenshtein(self, x, y):
        """Levenshtein distance between developers associated with entities

        Args:
            x (list): entity defining list
            y (list): entity defining list

        Returns:
            int: edit distance
        """
        x_developer = []
        for rel in x['relations']: 
            if rel['type'] == 'developer_of':
                x_developer.append(rel['string'])
        if not x_developer:
            return 100
        y_developer = []
        for rel in y['relations']: 
            if rel['type'] == 'developer_of':
                y_developer.append(rel['string'])
        if not y_developer:
            return 100
        min_dist = 100
        for u_x in x_developer: 
            for u_y in y_developer:
                dist = levenshtein_distance(u_x, u_y)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _developer_substring(self, x, y):
        """Developer substring relation between urls associated with entities

        Args:
            x (list): entity defining list
            y (list): entity defining list

        Returns:
            bool: result
        """
        x_developer = []
        for rel in x['relations']: 
            if rel['type'] == 'developer_of':
                x_developer.append(rel['string'])
        if not x_developer:
            return False
        y_developer = []
        for rel in y['relations']: 
            if rel['type'] == 'developer_of':
                y_developer.append(rel['string'])
        if not y_developer:
            return False
        for u_x in x_developer: 
            for u_y in y_developer:
                if u_x in u_y or u_y in u_x:
                    return True
        return False

    def _developer_substring_pos(self, x, y):
        """Developer substring relation between urls associated with entities.
        Empty URL count as positive matches.

        Args:
            x (list): entity defining list
            y (list): entity defining list

        Returns:
            bool: result
        """
        x_developer = []
        for rel in x['relations']: 
            if rel['type'] == 'developer_of':
                x_developer.append(rel['string'])
        if not x_developer:
            return True
        y_developer = []
        for rel in y['relations']: 
            if rel['type'] == 'developer_of':
                y_developer.append(rel['string'])
        if not y_developer:
            return True
        for u_x in x_developer: 
            for u_y in y_developer:
                if u_x in u_y or u_y in u_x:
                    return True
        return False

    def _abbreviation(self, x, y): 
        """Abbreviation relalation between entity mentions

        Args:
            x (list): entity defining list
            y (list): entity defining list

        Returns:
            bool: result
        """
        norm_x = re.sub('[^0-9a-zA-Z]+', ' ', x['mention'].casefold()).rstrip('0123456789 ,.').lstrip(' ')
        norm_x = ' '.join([w for w in norm_x.split() if w not in self.stops])
        abbr_x = ''.join([s[0] for s in norm_x.split()]) if len(norm_x.split()) > 2 else None
        norm_y = re.sub('[^0-9a-zA-Z]+', ' ', y['mention'].casefold()).rstrip('0123456789 ,.').lstrip(' ')
        norm_y = ' '.join([w for w in norm_y.split() if w not in self.stops])
        abbr_y = ''.join([s[0] for s in norm_y.split()]) if len(norm_y.split()) > 2 else None
        if abbr_y is not None and abbr_x is not None and ( abbr_x == abbr_y or abbr_x == y['mention'].lower() or abbr_y == x['mention'].lower() ):
            return True
        elif abbr_y is not None and abbr_y == x['mention'].lower():
            return True
        elif abbr_x is not None and abbr_x == y['mention'].lower():
            return True
        else:
            return False
        
    def _dbpedia_alt_names(self, x, y):
        """Test if entities are DBpedia altNames of each other

        Args:
            x (list): entity defining list
            y (list): entity defining list

        Returns:
            bool: result
        """
        x_unique = set()
        x_labels = set()
        if x['mention'] in self.dbpedia_unique_mapping:
            x_unique = copy.deepcopy(self.dbpedia_unique_mapping[x['mention']])
        if x['mention'] in self.dbpedia_label_mapping:
            x_labels = copy.deepcopy(self.dbpedia_label_mapping[x['mention']])
        x_labels.update(x_unique)
        if not bool(x_labels):
            return False
        y_unique = set()
        y_labels = set()
        if y['mention'] in self.dbpedia_unique_mapping:
            y_unique = copy.deepcopy(self.dbpedia_unique_mapping[y['mention']])
        if y['mention'] in self.dbpedia_label_mapping:
            y_labels = copy.deepcopy(self.dbpedia_label_mapping[y['mention']])
        y_labels.update(y_unique)
        if not bool(y_labels):
            return False
        return bool(x_labels & y_labels)

    def save_triangular_matrix(self, tri_matrix, dim, feature_name):
        """ This function saves a triangular matrix in a flat representation so unnecessary zeros can be 
        excluded. To implement it only the necessary indicies from the matrix are extracted line by line. 
        
        Arguments:
            tri_matrix (np.matrix with an arbitrary d_type): the matrix to save
            dim {length of the (quadratic) matrix): used to generate the required indicies
            feature_name (string): what the output file will be called
        """
        indices = np.triu_indices(dim, 1)
        flat_representation = tri_matrix[indices]
        np.save(join('feature_outputs', feature_name), flat_representation)

    def apply_features(self, entities):
        """ Applies an arbitrary number of features onto a list of entity names.
        The feature is calculated once for each possible pair of entity names. 
        
        Arguments:
            entities (list): list of entity names
        """
        matrices = {}
        for fct in self.features_to_extract:
            start_time = time.time()
            result_matrix = np.full((len(entities), len(entities)), fct['init'], dtype=fct['data_type'])
            list_to_iterate = entities.copy()
            #get matrix with rows and columns
            while list_to_iterate:
                cur_name = list_to_iterate.pop(-1)
                matrix_idx_y = len(list_to_iterate)
                for matrix_idx_x, name_to_compare in enumerate(list_to_iterate):
                    distance = fct['function'](cur_name, name_to_compare)
                    result_matrix[matrix_idx_x, matrix_idx_y] = distance
            matrices[fct['name']] = result_matrix
                
            end_time = time.time()
            print("It took {} seconds to calculate feature {} for {} inputs.".format(round(end_time-start_time, 3), fct['name'], len(entities)))
        return matrices
