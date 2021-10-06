import re
import numpy as np

from gensim.models import KeyedVectors
from itertools import combinations
from articlenizer import CUSTOM_STOPWORDS
from articlenizer.formatting import bio_to_brat

class FeatureGenerator():
    def __init__(self, data_handler, embedding_location, emb_dim=200):
        self.main_entities = ['Application_Creation', 'Application_Deposition', 'Application_Usage', 'Application_Mention', 'PlugIn_Creation', 'PlugIn_Deposition', 'PlugIn_Usage', 'PlugIn_Mention', 'ProgrammingEnvironment_Usage', 'ProgrammingEnvironment_Mention', 'OperatingSystem_Usage', 'OperatingSystem_Mention', 'SoftwareCoreference_Deposition']
        self.data_handler = data_handler
        self.type2idx, self.type2name, self.mention2idx, self.mention2name = {}, {}, {}, {}
        for k in sorted(data_handler.encoding['tag2name']):
            if '_' in data_handler.encoding['tag2name'][k]:
                ent_type, mention_type = data_handler.encoding['tag2name'][k].split('_')
            else:
                ent_type = data_handler.encoding['tag2name'][k]
                mention_type = 'None'
            if ent_type not in self.type2idx:
                self.type2name[len(self.type2idx)] = ent_type
                self.type2idx[ent_type] = len(self.type2idx)
            if mention_type not in self.mention2idx:
                self.mention2name[len(self.mention2idx)] = mention_type
                self.mention2idx[mention_type] = len(self.mention2idx)
        self.default_ent_type = 'Application'
        self.default_men_type = 'Usage'

        # self.word_emb = KeyedVectors.load_word2vec_format(embedding_location, binary=True)
        # self.emb_dim = emb_dim

    def correct_types(self, ent_0, ent_1, men_0, men_1):
        ent_0 = self.default_ent_type if ent_0 == 'Unknown' else ent_0
        ent_1 = self.default_ent_type if ent_1 == 'Unknown' else ent_1
        men_0 = self.default_men_type if men_0 == 'Unknown' else men_0
        men_1 = self.default_men_type if men_1 == 'Unknown' else men_1
        return ent_0, ent_1, men_0, men_1
        
    def acronym(self, tokens_in):
        """Get acronym of tokens

        Args:
            tokens_in (list): list of tokens

        Returns:
            str: acronym
        """
        tokens = [x.casefold() for x in tokens_in]
        tokens = [re.sub('[^0-9a-zA-Z]+', ' ', x) for x in tokens]
        tokens = [x.rstrip('0123456789 ,.').lstrip(' ') for x in tokens]
        tokens = [x for x in tokens if x not in CUSTOM_STOPWORDS]
        tokens = [x for x in tokens if x]
        if len(tokens) <= 2:
            return None
        acronym = ''.join([x[0] for x in tokens])
        return acronym

    def is_substring(self, e1, e2):
        """Substring relation between entities

        Args:
            e1 (str): entity string
            e2 (str): entity string

        Returns:
            bool: result
        """
        if e1 is None or e2 is None:
            return False
        return ''.join(e1).casefold() in ''.join(e2).casefold()

    def get_left_context_word(self, sentence, position):
        """Get word left of position in sentence

        Args:
            sentence (list): list of string tokens
            position (int): position to extract

        Returns:
            str: string of token left of position
        """
        sub_sent = sentence[:position]
        sub_tokens = sub_sent.split()
        if len(sub_tokens) > 0:
            return sub_tokens[-1]
        else:
            return None
    
    def get_right_context_word(self, sentence, position):
        """Get word right of position in sentence

        Args:
            sentence (list): list of string tokens
            position (int): position to extract

        Returns:
            str: string of token left of position
        """
        sub_sent = sentence[position:]
        sub_tokens = sub_sent.split()
        if len(sub_tokens) > 0:
            return sub_tokens[0]
        else:
            return None

    def one_hot_encoding(self, dictionary, name, ent_type, encoding):
        """Generate a one-hot encoding 

        Args:
            dictionary (dict): dictionary with information, is written and changed
            name (str): addition to output feature name
            ent_type ([type]): restriction for entries to consider
            encoding ([type]): mapping for names
        """
        found_key = False
        for k, v in encoding.items():
            feature_name = '{}_{}'.format(name, k)
            if ent_type == k:
                dictionary[feature_name] = 1
                found_key = True
            else:
                dictionary[feature_name] = 0
        if not found_key:
            raise(RuntimeError("Unknown entity type: {}".format(ent_type)))

    # def get_entities_inbetween(self, tags, ind1, ind2):
    #     """Get number of tagged elements inbetween two indicies

    #     Args:
    #         tags (list): sequence of tags
    #         ind1 (int): index
    #         ind2 (int): index

    #     Returns:
    #         int: number of tagged entities
    #     """
    #     sub_tags = tags[ind1:ind2].split()
    #     return sum([1 if x.startswith('B-') else 0 for x in sub_tags])

    def get_features(self, pair, sentence, tags, main_entity_count):
        """Get feature dictionary for pair of entities

        Args:
            pair (tuple): pair of entities
            sentence (list): sentence containing entities as tokens
            tags (list): IOB2 labels assigned to sentence
            main_entity_count (int): number of main entities in sentence
            entity_encoding (dictionary): numeric encoding of entities

        Returns:
            dictionary: features for pair
        """
        if pair[0]['beg'] > pair[1]['beg']:
            larger = pair[0]
            smaller = pair[1]
        else:
            larger = pair[1]
            smaller = pair[0]
        distance_string = sentence[smaller['end']:larger['beg']]
        tokens_distance_string = distance_string.split()

        # ent_0_left_context = self.get_left_context_word(sentence, pair[0]['beg'])
        # ent_1_left_context = self.get_left_context_word(sentence, pair[1]['beg'])
        # ent_0_right_context = self.get_right_context_word(sentence, pair[0]['end'])
        # ent_1_right_context = self.get_right_context_word(sentence, pair[1]['end'])
        ent_0_tokens = pair[0]['string'].split()
        ent_1_tokens = pair[1]['string'].split()
        ent_0_acronym = self.acronym(ent_0_tokens)
        ent_1_acronym = self.acronym(ent_1_tokens)

        #entities_in_between = self.get_entities_inbetween(tags, smaller['end'], larger['beg'])
        
        # if ent_0_left_context in self.word_emb:
        #     ent_0_left_context_emb = self.word_emb[ent_0_left_context]
        # else:
        #     ent_0_left_context_emb = np.zeros(self.emb_dim)
        # if ent_1_left_context in self.word_emb:
        #     ent_1_left_context_emb = self.word_emb[ent_1_left_context]
        # else:
        #     ent_1_left_context_emb = np.zeros(self.emb_dim)
        # if ent_0_right_context in self.word_emb:
        #     ent_0_right_context_emb = self.word_emb[ent_0_right_context]
        # else:
        #     ent_0_right_context_emb = np.zeros(self.emb_dim)
        # if ent_1_right_context in self.word_emb:
        #     ent_1_right_context_emb = self.word_emb[ent_1_right_context]
        # else:
        #     ent_1_right_context_emb = np.zeros(self.emb_dim)

        features = {
            'entity_distance_abs': len(distance_string), 
            ###'entity_distance_rel': len(distance_string) / len(sentence), 
            'entity_distance_tok': len(tokens_distance_string), 
            ###'entity_distance_tok_rel': len(tokens_distance_string) / len(sentence.split()), 
            'entity_order': pair[1]['beg'] > pair[0]['beg'], 
            'entity_0_char_length': pair[0]['end'] - pair[0]['beg'],
            'entity_0_token_length': len(ent_0_tokens),
            'entity_1_char_length': pair[1]['end'] - pair[1]['beg'],
            'entity_1_token_length': len(ent_1_tokens),
            ###'entity_0_left_context_for': ent_0_left_context == 'for', 
            ###'entity_1_left_context_for': ent_1_left_context == 'for', 
            #'ent_0_left_context_emb': ent_0_left_context_emb,
            #'ent_1_left_context_emb': ent_1_left_context_emb,
            #'ent_0_right_context_emb': ent_0_right_context_emb,
            #'ent_1_right_context_emb': ent_1_right_context_emb,
            'num_main_entities': main_entity_count,
            'entity_0_substring_of_entity_1': self.is_substring(pair[0]['string'], pair[1]['string']),
            'acronym_entity_0_substring_of_entity_1': self.is_substring(ent_0_acronym, pair[1]['string']), 
            'acronym_entity_0_substring_of_acronym_entity_1': self.is_substring(ent_0_acronym, ent_1_acronym),
            'entity_1_substring_of_entity_0': self.is_substring(pair[1]['string'], pair[0]['string']),
            'acronym_entity_1_substring_of_entity_0': self.is_substring(ent_1_acronym, pair[0]['string']), 
            'acronym_entity_1_substring_of_acronym_entity_0': self.is_substring(ent_1_acronym, ent_0_acronym)
        }

        ent_0_type = pair[0]['type'].split('_')[0]
        mention_0_type = pair[0]['type'].split('_')[1] if '_' in pair[0]['type'] else 'None'
        ent_1_type = pair[1]['type'].split('_')[0]
        mention_1_type = pair[1]['type'].split('_')[1] if '_' in pair[1]['type'] else 'None'
        ent_0_type, ent_1_type, mention_0_type, mention_1_type = self.correct_types(ent_0_type, ent_1_type, mention_0_type, mention_1_type)
        self.one_hot_encoding(features, 'entity_0_type', ent_0_type, self.type2idx)
        self.one_hot_encoding(features, 'mention_0_type', mention_0_type, self.mention2idx)
        self.one_hot_encoding(features, 'entity_1_type', ent_1_type, self.type2idx)
        self.one_hot_encoding(features, 'mention_1_type', mention_1_type, self.mention2idx)

        # for idx, (x, y, z, a) in enumerate(zip(ent_0_left_context_emb, ent_1_left_context_emb, ent_0_right_context_emb, ent_1_right_context_emb)):
        #     features['ent_0_left_context_emb_{}'.format(idx)] = x
        #     features['ent_1_left_context_emb_{}'.format(idx)] = y
        #     features['ent_0_right_context_emb_{}'.format(idx)] = z
        #     features['ent_1_right_context_emb_{}'.format(idx)] = a
        return features


    def get_sentence_relations_and_features(self, sentence, tags, entities, relations=None):
        """Span all potential relations in sentence and calculate its respective features

        Args:
            sentence (list): sentence as token list
            tags (list): IOB2 labels assigned to sentence
            entities (dict): entities in sentence
            relations (dict): true relations in sentence

        Returns:
            list: relations in sentence including features and labels
        """
        sentence_feature_list = []
        
        entities = sorted(entities, key=lambda item: item['beg'])
        for idx, x in enumerate(entities):
            x['idx'] = idx
            # TODO add this index as training data?

        main_entity_count = 0
        for ent in entities:
            if ent['type'] in self.main_entities:
                main_entity_count += 1

        entity_combinations = combinations(entities, 2)
        entity_pairs = []
        for comb in entity_combinations:
            entity_pairs.extend([list(comb), list(reversed(comb))])

        assigned_count = 0
        for idx, pair in enumerate(entity_pairs):
            features = self.get_features(pair, sentence, tags, main_entity_count)
            if relations is not None:
                features['label'] = 'none'
                for relation in relations:
                    if int(relation['ent1_b']) == int(pair[0]['beg']) and int(relation['ent2_b']) == int(pair[1]['beg']):
                        features['label'] = relation['type']
                        assigned_count += 1
                        break
            sentence_feature_list.append(features)

        return sentence_feature_list, entity_pairs 

    def generate_relation_extraction_features(self):
        for dataset, dataset_setup in self.data_handler.data_config['sets'].items():
            for sub_dataset in dataset_setup:
                sub_dataset['relext_feature_list'] = []
                for sentence, tags, relations in zip(sub_dataset['sentences'], sub_dataset['tags'], sub_dataset['relations']):
                    if sum([1 if t.startswith('B-') else 0 for t in tags.split()]) > 1: 
                        entities, _, _ = bio_to_brat(sentence, tags)
                        if self.data_handler.tag_remapping is not None:
                            for ent in entities:
                                ent['type'] = self.data_handler.tag_remapping[ent['type']]
                        pairs, _ = self.get_sentence_relations_and_features(sentence, tags, entities, relations)
                        sub_dataset['relext_feature_list'].extend(pairs)

    def stream_files(self):
        for file_names in self.data_handler.data_files:
            article = {
                'out_name': file_names['out'],
                'sentences': self.data_handler._read_text_file(file_names['in']),
                'tags': self.data_handler._read_text_file(file_names['entities']),
                'entity_list': [],
                'relext_feature_list': []
            }
            for idx, (sentence, tags) in enumerate(zip(article['sentences'], article['tags'])):
                if sum([1 if t.startswith('B-') else 0 for t in tags.split()]) > 1: 
                    entities, _, _ = bio_to_brat(sentence, tags)
                    if self.data_handler.tag_remapping is not None:
                        for ent in entities:
                            ent['type'] = self.data_handler.tag_remapping[ent['type']]
                    pairs, entity_pairs = self.get_sentence_relations_and_features(sentence, tags, entities)
                    article['entity_list'].append(list(entity_pairs))
                    article['relext_feature_list'].append(pairs)
                else:
                    article['entity_list'].append(None)
                    article['relext_feature_list'].append(None)

            yield article
