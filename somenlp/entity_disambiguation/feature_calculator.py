import numpy as np
import pandas as pd
import time
import re
import torch
import json
import nltk
nltk.download('stopwords')

from itertools import combinations
from nltk.corpus import stopwords
from multiprocessing import Pool

from os.path import join, exists
from Levenshtein import distance as levenshtein_distance
from Levenshtein import jaro

from articlenizer import articlenizer

MENTION_SCALING_FACTOR = 60
URL_SCALING_FACTOR = 120
DEVEL_SCALING_FACTOR = 30
VERSION_SCALING_FACTOR = 10

class EntityDisambiguationFeatureGenerator:
    """Calculates features for entity disambiguation
    """
    def __init__(self, dbpedia):
        """Init

        Args:
            dbpedia (str): location of a pandas dataframe containing DBpedia information for software entities
        """
        self.stops = stopwords.words('english')
        # self.dbpedia_names = pd.read_csv(dbpedia, compression='gzip')
        # self.dbpedia_names.drop(self.dbpedia_names.columns.difference(['unique','label']), 1, inplace=True)
        # self.dbpedia_names.dropna(inplace=True)
        # self.dbpedia_unique_mapping = {}
        # self.dbpedia_label_mapping = {}
        # for index, row in self.dbpedia_names.iterrows():
        #     if row['unique'] not in self.dbpedia_unique_mapping:
        #         self.dbpedia_unique_mapping[row['unique']] = set()
        #     self.dbpedia_unique_mapping[row['unique']].update([row['label']])
        #     if row['label'] not in self.dbpedia_label_mapping:
        #         self.dbpedia_label_mapping[row['label']] = set()
        #     self.dbpedia_label_mapping[row['label']].update([row['unique']])
        with open(dbpedia, 'r') as j_in:
            self.dbpedia_data = json.load(j_in)
        #self.dbpedia_unique_mapping = dbpedia_data['unique_mapping']
        #self.dbpedia_label_mapping = dbpedia_data['label_mapping']

        self.string_features_to_extract = [
            { # String based features
                'name': 'LenFirst',
                'function': self._len_first,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'LenSecond',
                'function': self._len_second,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'Jaro',
                'function': self._jaro,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'Levenshtein',
                'function': self._levenshtein,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'Substring',
                'function': self._substring,
                'data_type': np.float32,
                'init': 1.0
            },
            { 
                'name': 'Norm_string_Jaro',
                'function': self._norm_string_jaro,
                'data_type': np.float32,
                'init': 1.0
            },
            { 
                'name': 'Norm_string_Levenshtein',
                'function': self._norm_string_levenshtein,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'KnownAbbr',
                'function': self._known_abbreviation,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'MenGenAbbr',
                'function': self._mention_generated_abbreviation,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'NormGenAbbr',
                'function': self._norm_generated_abbreviation,
                'data_type': np.float32,
                'init': 1.0
            }
        ]
        self.context_features_to_extract = [
            { # Abbreviations and Alternative Names
                'name': 'GivenAbbr',
                'function': self._given_abbreviation,
                'data_type': np.float32,
                'init': 1.0
            },
            { # URLs
                'name': 'URL_LenFirst',
                'function': self._url_len_first,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'URL_LenSecond',
                'function': self._url_len_second,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'URL_Jaro',
                'function': self._url_jaro,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'URL_Substring',
                'function': self._url_substring,
                'data_type': np.float32,
                'init': 1.0
            },
            { # Developers
                'name': 'Devel_LenFirst',
                'function': self._devel_len_first,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'Devel_LenSecond',
                'function': self._devel_len_second,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'Devel_Jaro',
                'function': self._devel_jaro,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'Devel_Substring',
                'function': self._devel_substring,
                'data_type': np.float32,
                'init': 1.0
            },
            { # Versions
                'name': 'Version_LenFirst',
                'function': self._version_len_first,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'Version_LenSecond',
                'function': self._version_len_second,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'Version_Jaro',
                'function': self._version_jaro,
                'data_type': np.float32,
                'init': 1.0
            },
            {
                'name': 'Version_Substring',
                'function': self._version_substring,
                'data_type': np.float32,
                'init': 1.0
            }
        ]
        self.feature_length = len(self.string_features_to_extract) + len(self.context_features_to_extract)

    def _len_fct(self, s, factor):
        return len(s) / factor

    def _jaro_fct(self, s0, s1):
        return 1 - jaro(s0, s1)

    def _levenshtein_fct(self, s0, s1):
        return levenshtein_distance(s0, s1) / max(len(s0), len(s1))

    def _substring_fct(self, s0, s1):
        return 1 - (s0 in s1 or s1 in s0)

    def _len_first(self, pair):
        return self._len_fct(pair[0]['string'], MENTION_SCALING_FACTOR)
    
    def _len_second(self, pair):
        return self._len_fct(pair[1]['string'], MENTION_SCALING_FACTOR)

    def _jaro(self, pair):
        """Jaro distance between entity mentions

        Args:
            pair (list): entity pair
        """
        return self._jaro_fct(pair[0]['string'], pair[1]['string'])

    def _levenshtein(self, pair):
        """Levenshtein distance between entity mentions

        Args:
            pair (list): entity pair

        Returns:
            int: Levenshtein edit distance
        """
        return self._levenshtein_fct(pair[0]['string'], pair[1]['string'])

    def _substring(self, pair):
        """Test substring relation between entity mentions

        Args:
            pair (list): entity pair

        Returns:
            bool: result
        """
        return self._substring_fct(pair[0]['string'], pair[1]['string'])

    def _normalize(self, s):
        norm_s = re.sub('[^0-9a-zA-Z]+', ' ', s.casefold()).rstrip('0123456789 ,.').lstrip(' ')
        norm_s = ' '.join([w for w in norm_s.split() if w not in self.stops])
        if not norm_s:
            norm_s = s
        return norm_s

    def _remove_spaces(self, s):
        replace_regex = re.compile(r'\s(?P<to_keep>[\+\-#™_/\d]+)\s?')
        matches = replace_regex.findall(s)
        return replace_regex.sub(r'\g<to_keep>', s)

    def _norm_string_jaro(self, pair):
        return self._jaro_fct(pair[0]['norm'], pair[1]['norm'])

    def _norm_string_levenshtein(self, pair):
        return self._levenshtein_fct(pair[0]['norm'], pair[1]['norm'])

    def _add_info_len(self, pair, idx, name, factor):
        num = 0
        length = 0
        for rel in pair[idx]['relations']: 
            if rel['type'] == name:
                num += 1
                length += len(rel['string'])
        length = length / num if num > 0 else 0
        return length / factor

    def _add_info_jaro(self, pair, name):
        x_infos = []
        for rel in pair[0]['relations']: 
            if rel['type'] == name:
                x_infos.append(rel['string'])
        if not x_infos:
            return 1.0
        y_infos = []
        for rel in pair[1]['relations']: 
            if rel['type'] == name:
                y_infos.append(rel['string'])
        if not y_infos:
            return 1.0
        min_dist = 1.0
        for i_x in x_infos: 
            for i_y in y_infos:
                dist = self._jaro_fct(i_x, i_y)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _add_info_substring(self, pair, name):
        x_infos = []
        for rel in pair[0]['relations']: 
            if rel['type'] == name:
                x_infos.append(rel['string'])
        if not x_infos:
            return 1
        y_infos = []
        for rel in pair[1]['relations']: 
            if rel['type'] == name:
                y_infos.append(rel['string'])
        if not y_infos:
            return 1
        for i_x in x_infos: 
            for i_y in y_infos:
                if i_x in i_y or i_y in i_x:
                    return 0
        return 1

    def _url_len_first(self, pair):
        return self._add_info_len(pair, 0, 'URL_of', URL_SCALING_FACTOR)

    def _url_len_second(self, pair):
        return self._add_info_len(pair, 1, 'URL_of', URL_SCALING_FACTOR)

    def _url_jaro(self, pair):
        return self._add_info_jaro(pair, 'URL_of')

    def _url_substring(self, pair):
        return self._add_info_substring(pair, 'URL_of')

    def _devel_len_first(self, pair):
        return self._add_info_len(pair, 0, 'Developer_of', DEVEL_SCALING_FACTOR)

    def _devel_len_second(self, pair):
        return self._add_info_len(pair, 1, 'Developer_of', DEVEL_SCALING_FACTOR)

    def _devel_jaro(self, pair):
        return self._add_info_jaro(pair, 'Developer_of')

    def _devel_substring(self, pair):
        return self._add_info_substring(pair, 'Developer_of')

    def _version_len_first(self, pair):
        return self._add_info_len(pair, 0, 'Version_of', VERSION_SCALING_FACTOR)

    def _version_len_second(self, pair):
        return self._add_info_len(pair, 1, 'Version_of', VERSION_SCALING_FACTOR)

    def _version_jaro(self, pair):
        return self._add_info_jaro(pair, 'Version_of')

    def _version_substring(self, pair):
        return self._add_info_substring(pair, 'Version_of')

    def _mention_generated_abbreviation(self, pair):
        men_str_x = pair[0]['string'].replace('-', ' ')
        men_abbr_x = ''.join([s[0] for s in men_str_x.split()]) if len(men_str_x.split()) > 2 else ''
        men_str_y = pair[1]['string'].replace('-', ' ')
        men_abbr_y = ''.join([s[0] for s in men_str_y.split()]) if len(men_str_y.split()) > 2 else ''
        if not men_abbr_x and not men_abbr_y:
            return 1.0
        min_dist = min(self._jaro_fct(men_abbr_x, pair[1]['string']), self._jaro_fct(men_abbr_y, pair[0]['string']), self._jaro_fct(men_abbr_x, men_abbr_y))
        return min_dist

    def _norm_generated_abbreviation(self, pair):
        men_str_x = pair[0]['norm'].replace('-', ' ')
        men_abbr_x = ''.join([s[0] for s in men_str_x.split()]) if len(men_str_x.split()) > 2 else ''
        men_str_y = pair[1]['norm'].replace('-', ' ')
        men_abbr_y = ''.join([s[0] for s in men_str_y.split()]) if len(men_str_y.split()) > 2 else ''
        if not men_abbr_x and not men_abbr_y:
            return 1.0
        min_dist = min(self._jaro_fct(men_abbr_x, pair[1]['norm']), self._jaro_fct(men_abbr_y, pair[0]['norm']), self._jaro_fct(men_abbr_x, men_abbr_y))
        return min_dist

    def _known_abbreviation(self, pair):
        x_altnames = set([pair[0]['string']])
        if pair[0]['string'] in self.dbpedia_data['unique_mapping']:
            x_altnames.update(self.dbpedia_data['unique_mapping'][pair[0]['string']])
        elif pair[0]['string'] in self.dbpedia_data['label_mapping']:
            for key in self.dbpedia_data['label_mapping'][pair[0]['string']]:
                x_altnames.update([key])
                x_altnames.update(self.dbpedia_data['unique_mapping'][key])
        y_altnames = set([pair[1]['string']])
        if pair[1]['string'] in self.dbpedia_data['unique_mapping']:
            y_altnames.update(self.dbpedia_data['unique_mapping'][pair[1]['string']])
        elif pair[1]['string'] in self.dbpedia_data['label_mapping']:
            for key in self.dbpedia_data['label_mapping'][pair[1]['string']]:
                y_altnames.update([key])
                y_altnames.update(self.dbpedia_data['unique_mapping'][key])
        min_dist = 1.0
        for i_x in x_altnames:
            dist = self._jaro_fct(i_x, pair[1]['string'])
            if dist < min_dist:
                min_dist = dist
        for i_y in y_altnames:
            dist = self._jaro_fct(i_y, pair[0]['string'])
            if dist < min_dist:
                min_dist = dist
        for i_x in x_altnames: 
            for i_y in y_altnames:
                dist = self._jaro_fct(i_x, i_y)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _given_abbreviation(self, pair):
        x_infos = []
        for rel in pair[0]['relations']: 
            if rel['type'] in ['Abbreviation_of', 'AlternativeName_of']:
                x_infos.append(rel['string'])
        y_infos = []
        for rel in pair[1]['relations']: 
            if rel['type'] in ['Abbreviation_of', 'AlternativeName_of']:
                y_infos.append(rel['string'])
        min_dist = 1.0
        for i_x in x_infos:
            dist = self._jaro_fct(i_x, pair[1]['string'])
            if dist < min_dist:
                min_dist = dist
        for i_y in y_infos:
            dist = self._jaro_fct(i_y, pair[0]['string'])
            if dist < min_dist:
                min_dist = dist
        for i_x in x_infos: 
            for i_y in y_infos:
                dist = self._jaro_fct(i_x, i_y)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def features_for_pair(self, pair):
        results = []
        for fct in self.string_features_to_extract:
            result = fct['function'](pair)
            results.append(result)
        for fct in self.context_features_to_extract:
            result = fct['function'](pair)
            results.append(result)
        #print(results)
        #print(torch.tensor(result).data)
        return results

    def features_for_pre_clustered(self, pair):
        string_results = []
        #string_results = np.empty((len(self.string_features_to_extract)))
        for fct in self.string_features_to_extract:
            result = fct['function'](pair)
            string_results.append(result)
            #string_results[idx] = result
        context_results = []
        #context_results = np.empty((len(pair[0]['contexts'])*len(pair[1]['contexts']), len(self.context_features_to_extract)))
        #count = 0
        for context_first in pair[0]['contexts']:
            context_first['string'] = pair[0]['string']
            for context_second in pair[1]['contexts']:
                context_second['string'] = pair[1]['string']
                context_results.append(string_results.copy())
                for fct in self.context_features_to_extract:
                    result = fct['function']([context_first, context_second])
                    context_results[-1].append(result)
                    #context_results[count, idx] += result
                #count += 1
        return context_results#string_results, context_results

    def apply_features(self, entities, ncores=5):
        """ Applies an arbitrary number of features onto a list of entity names.
        The feature is calculated once for each possible pair of entity names. 
        
        Arguments:
            entities (list): list of entity names
        """
        matrices = {}
        for entity in entities:
            entity['string'] = self._remove_spaces(entity['mention'])
            entity['norm'] = self._normalize(entity['mention'])
        pairs_to_compare = list(combinations(entities, 2))
        # for fct in self.features_to_extract:
        #     start_time = time.time()
        #     if ncores > 1:
        #         with Pool(ncores) as p:
        #             result = p.map(fct['function'], pairs_to_compare)
        #         matrices[fct['name']] = result
        #     else:
        #         result = np.full((sum(range(len(entities)))), fct['init'], dtype=fct['data_type'])
        #         for idx, pair in enumerate(pairs_to_compare):
        #             distance = fct['function'](pair)
        #             result[idx] = distance
        #         matrices[fct['name']] = result      
        #     end_time = time.time()
        #     print("It took {} seconds to calculate feature {} for {} inputs.".format(round(end_time-start_time, 3), fct['name'], len(entities)))
        for fct in self.string_features_to_extract:
            start_time = time.time()
            if ncores > 1:
                with Pool(ncores) as p:
                    result = p.map(fct['function'], pairs_to_compare)
                matrices[fct['name']] = result
            else:
                result = np.full((sum(range(len(entities)))), fct['init'], dtype=fct['data_type'])
                for idx, pair in enumerate(pairs_to_compare):
                    distance = fct['function'](pair)
                    result[idx] = distance
                matrices[fct['name']] = result      
            end_time = time.time()
            print("It took {} seconds to calculate feature {} for {} inputs.".format(round(end_time-start_time, 3), fct['name'], len(entities)))
        for fct in self.context_features_to_extract:
            start_time = time.time()
            if ncores > 1:
                with Pool(ncores) as p:
                    result = p.map(fct['function'], pairs_to_compare)
                matrices[fct['name']] = result
            else:
                result = np.full((sum(range(len(entities)))), fct['init'], dtype=fct['data_type'])
                for idx, pair in enumerate(pairs_to_compare):
                    distance = fct['function'](pair)
                    result[idx] = distance
                matrices[fct['name']] = result      
            end_time = time.time()
            print("It took {} seconds to calculate feature {} for {} inputs.".format(round(end_time-start_time, 3), fct['name'], len(entities)))
        return matrices

    def get_labels(self, entities, gold):
        for g in gold:
            g_sentences = articlenizer.get_tokenized_sentences(g['sentence'])
            g['prosentence'] = [' '.join(s) for s in g_sentences]
            g['proconcat'] = ' '.join(g['prosentence'])
        for ent in entities:
            for idx, g in enumerate(gold):
                if ent['paper_id'] == g['paper_id'] and any([ent['sentence'] == s for s in g['prosentence']]) or ent['sentence'] == g['proconcat']:
                    if ent['mention'] == ' '.join(articlenizer.get_tokenized_sentences(g['mention'])[0]):
                        ent['gold_id'] = idx
                        break
        pairs_to_compare = list(combinations(entities, 2))
        start_time = time.time()
        result = np.full((sum(range(len(entities)))), 0, dtype=np.int16)
        for idx, pair in enumerate(pairs_to_compare):
            if 'gold_id' in pair[0] and 'gold_id' in pair[1] and gold[pair[0]['gold_id']]['link'] == gold[pair[1]['gold_id']]['link']:
                result[idx] = 1
        end_time = time.time()
        print("It took {} seconds to get labels for {} inputs.".format(round(end_time-start_time, 3), len(entities)))
        return result
