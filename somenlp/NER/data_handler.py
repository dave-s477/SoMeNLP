import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
import random
import copy

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from pathlib import Path
from gensim.models import KeyedVectors
from itertools import zip_longest
from articlenizer import articlenizer as art
from transformers import BertTokenizer

from .LSTM_dataset import LSTMDataset
from .BERT_dataset import BERTDataset, BERTMultiDataset

BERT_MAX_LENGTH = 256

class DataHandler():
    def __init__(self, data_config=None, data_file_extension='.data.txt', label_file_extension='.labels.txt', feature_file_extension='', relation_file_extension='', output_handler=None, checkpoint=None, padding='<PAD>', unknown='<UNK>', batch_size=32, max_word_length=-1, max_sent_length=-1, data_files=None, prepro=False, tokenizer=None, multi_task=False):
        self.data_config = copy.deepcopy(data_config)
        self.data_file_extension = data_file_extension
        self.label_file_extension = label_file_extension
        self.feature_file_extension = feature_file_extension
        self.relation_file_extension = relation_file_extension
        self.checkpoint = checkpoint
        self.padding = padding
        self.unknown = unknown
        self.output_handler = output_handler
        self.batch_size = batch_size
        self.max_word_length = max_word_length
        self.max_sent_length = max_sent_length
        self.data_files = data_files
        self.prepro = prepro
        self.feature_dim = None
        self.multi_task_mapping = multi_task
        self.data = []
        self.features = []
        self.labels = []
        self._load_tag_remapping()
        self._load_relation_remapping()
        self.tokenizer = tokenizer
        if tokenizer is not None:
            self._setup_tokenizer(tokenizer)

    def _setup_tokenizer(self, location):
        self.tokenizer = BertTokenizer.from_pretrained(location, do_lower_case=False)
        self.padding = "[PAD]"
        self.special_toks = {
            'pad_tok': self.tokenizer.vocab["[PAD]"],
            'sep_tok': self.tokenizer.vocab["[SEP]"],
            'cls_tok': self.tokenizer.vocab["[CLS]"]
        }

    def _load_tag_remapping(self):
        if self.data_config is not None and 'transform' in self.data_config and 'mapping' in self.data_config['transform'] and self.data_config['transform']['mapping']:
            print("Loading tag remapping: {}".format(self.data_config['transform']['mapping']))
            mapping = Path(self.data_config['transform']['mapping'])
            with mapping.open(mode='r') as mapping_j:
                self.tag_remapping = json.load(mapping_j)
        elif self.data_config is not None and 'transform' in self.data_config and len(self.data_config['transform']) > 0:
            print("Loading tag mappings for a multi-label tagging problem")
            self.tag_remapping = {}
            self.multi_task_mapping = True
            for k, v in self.data_config['transform'].items():
                with Path(v).open(mode='r') as mapping_j:
                    self.tag_remapping[k] = json.load(mapping_j)
        else:
            self.tag_remapping = None

    def _load_relation_remapping(self):
        if self.data_config is not None and 'transform' in self.data_config and 'relation_mapping' in self.data_config['transform'] and self.data_config['transform']['relation_mapping']:
            print("Loading relation remapping: {}".format(self.data_config['transform']['relation_mapping']))
            mapping = Path(self.data_config['transform']['relation_mapping'])
            with mapping.open(mode='r') as mapping_j:
                self.relation_remapping = json.load(mapping_j)
        else:
            self.relation_remapping = None

    def _adjust_tag(self, tag, key=''):
        if self.tag_remapping is None or tag == 'O':
            return tag
        else:
            tag_prefix, tag_name = tag.split('-')
            if not key:
                remapped_tag = self.tag_remapping[tag_name]
            else:
                remapped_tag = self.tag_remapping[key][tag_name]
            if remapped_tag == 'O':
                return 'O'
            else:
                return '{}-{}'.format(tag_prefix, remapped_tag)

    def stream_files(self):
        for f_conf in self.data_files:
            if self.prepro:
                text = self._prepro_text_file(f_conf['in'])
            else:
                plain_text = self._read_text_file(f_conf['in'])
                text = []
                for line in plain_text:
                    text.append(line.split())
            if 'feat' in f_conf:
                features = self._read_feature_file(f_conf['feat'])
            else:
                features = []
            if self.tokenizer is None:
                characters, ids, tags, features = self._prepare_prediction_data(text, features)
                input_data = LSTMDataset(characters, ids, tags, features, self.encoding['char2idx'], self.padding, self.max_word_length, self.max_sent_length)
                sampler = SequentialSampler(input_data)
                data_loader = DataLoader(input_data, sampler=sampler, batch_size=self.batch_size, collate_fn=self._collate_fn)
            else:
                ids, tags, masks, lengths = self._prepare_bert_prediction_data(text)
                ids = self._pad_to_length(ids, length=BERT_MAX_LENGTH, fill_value=self.special_toks['pad_tok'], dtype=torch.long)
                masks = self._pad_to_length(masks, length=BERT_MAX_LENGTH, fill_value=0, dtype=torch.long)
                if not self.multi_task_mapping:
                    tags = self._pad_to_length(tags, length=BERT_MAX_LENGTH, fill_value=self.encoding['tag2idx']['O'], dtype=torch.long)
                    input_data = BERTDataset(ids, tags, masks)
                else:
                    for k in tags.keys():
                        tags[k] = self._pad_to_length(tags[k], length=BERT_MAX_LENGTH, fill_value=self.encoding['tag2idx'][k]['O'], dtype=torch.long)
                    input_data = BERTMultiDataset(ids, tags, masks, lengths)
                sampler = SequentialSampler(input_data)
                data_loader = DataLoader(input_data, sampler=sampler, batch_size=self.batch_size)
            
            yield [{'out': f_conf['out'], 'out-text': f_conf['out-text']}, data_loader, text]

    def load_data_from_config(self):
        for dataset, dataset_setup in self.data_config['sets'].items():
            for sub_dataset in dataset_setup:
                sub_dataset['all_files'] = []
                for folder in sub_dataset['folder']:
                    files = list(Path(folder).rglob('*{}'.format(self.data_file_extension)))
                    for f in files:
                        base_file_name = f.name.split(self.data_file_extension)[0]
                        sample_files = {}
                        if self.data_file_extension:
                            f_entry = Path('{}/{}{}'.format(str(f.parents[0]), base_file_name, self.data_file_extension))
                            sample_files[self.data_file_extension] = f_entry
                        if self.label_file_extension:
                            f_entry = Path('{}/{}{}'.format(str(f.parents[0]), base_file_name, self.label_file_extension))
                            if not f_entry.is_file():
                                raise(RuntimeError("Label file {} not found".format(f_entry)))
                            sample_files[self.label_file_extension] = f_entry
                        if self.feature_file_extension:
                            f_entry = Path('{}/{}{}'.format(str(f.parents[0]), base_file_name, self.feature_file_extension))
                            if not f_entry.is_file():
                                raise(RuntimeError("Feature file {} not found".format(f_entry)))
                            sample_files[self.feature_file_extension] = f_entry
                        if self.relation_file_extension:
                            f_entry = Path('{}/{}{}'.format(str(f.parents[0]), base_file_name, self.relation_file_extension))
                            if not f_entry.is_file():
                                raise(RuntimeError("Relation file {} not found".format(f_entry)))
                            sample_files[self.relation_file_extension] = f_entry
                        sub_dataset['all_files'].append(sample_files)

    def encoding(self, tags_only=False):
        if self.checkpoint is not None and self.checkpoint['model']:
            print("Loading given encodings")
            self.encoding = self.output_handler.load_encoding()
            for k, v in self.encoding.items():
                if self.multi_task_mapping and k.endswith('tag2name'):
                    for sk, vk in v.items():
                        vk_new = {int(key): value for key, value in vk.items()}
                        self.encoding[k][sk] = vk_new
                elif k.endswith('name'):
                    v_new = {int(key): value for key, value in v.items()}
                    self.encoding[k] = v_new
        else:
            print("Generating new encodings")

            # create separate dict for encodings for tagtoINdex, tagToname, wordToname, characterToName
            tag2idx, tag2name, word2idx, word2name, char2idx, char2name = {}, {}, {}, {}, {}, {}
            if not tags_only:
                for dataset, dataset_setup in self.data_config['sets'].items():
                    for sub_dataset in dataset_setup:
                        for f in sub_dataset['all_files']:
                                with f[self.data_file_extension].open(mode='r') as in_f:

                                    
                                    for line in in_f:

                                        # example 1st line in file : 'Animat brains consist of 8 binary elements ...' 
                                        #         2nd line in file : 'The sensors are directed upwards ...'
                                        for word in line.rstrip().split():

                                            # example: first word in first line: word = 'Animat'
                                            if word not in word2idx:
                                                # word2name => {0: 'Animat', 1: 'brains', 2: 'consist', 3: 'of', 4: '8', ..., 28:'The', 29:'are', 30:'directed', 31:'upwars', ...}
                                                word2name[len(word2idx)] = word

                                                # word2idx => {'Animat': 0, 'brains': 1, 'consist': 2, 'of': 3, '8': 4, ..., 'The':28, 'are':29, 'directed':30, 'upwards':31, ...}
                                                word2idx[word] = len(word2idx)

                                            for char in word:
                                                if char not in char2idx:

                                                    # char2name => {0: 'A', 1: 'n', 2: 'i', 3: 'm', 4: 'a', 5: 't', 6: 'b', 7: 'r', 8: 's', 9: 'c', 10: 'o', 11: 'f', 12: '8', 13: 'y', ...}
                                                    char2name[len(char2idx)] = char

                                                    # char2idx => {'A': 0, 'n': 1, 'i': 2, 'm': 3, 'a': 4, 't': 5, 'b': 6, 'r': 7, 's': 8, 'c': 9, 'o': 10, 'f': 11, '8': 12, 'y': 13, ...}
                                                    char2idx[char] = len(char2idx)
            if self.multi_task_mapping:
                print("Considering a multi-task problem")
                for dataset, dataset_setup in self.data_config['sets'].items():
                    for sub_dataset in dataset_setup:
                        for f in sub_dataset['all_files']:
                            with f[self.label_file_extension].open(mode='r') as in_f:
                                for line in in_f:
                                    for tag in line.rstrip().split():
                                        for mapping_key in self.tag_remapping.keys():
                                            if mapping_key not in tag2idx:
                                                tag2idx[mapping_key] = {}
                                                tag2name[mapping_key] = {}
                                            t = self._adjust_tag(tag, mapping_key)
                                            if tags_only and '-' in t:
                                                t = t.split('-')[-1]
                                            if t not in tag2idx[mapping_key]:
                                                tag2name[mapping_key][len(tag2idx[mapping_key])] = t
                                                tag2idx[mapping_key][t] = len(tag2idx[mapping_key])
                
                for k, v in tag2idx.items():
                    to_add = set()
                    for tk, vk in v.items():
                        if tk.startswith('B-') and 'I-' + tk.split('B-')[-1] not in v:
                            to_add.update(['I-' + tk.split('B-')[-1]])
                    for i in to_add:
                        tag2name[k][len(tag2idx[k])] = i
                        tag2idx[k][i] = len(tag2idx[k])
            else:
                print("considering a single-task problem")
                for dataset, dataset_setup in self.data_config['sets'].items():
                    for sub_dataset in dataset_setup:
                        for f in sub_dataset['all_files']:
                            with f[self.label_file_extension].open(mode='r') as in_f:
                                for line in in_f:
                                    for tag in line.rstrip().split():
                                        t = self._adjust_tag(tag)
                                        if tags_only and '-' in t:
                                            t = t.split('-')[-1]
                                        if t not in tag2idx:
                                            tag2name[len(tag2idx)] = t
                                            tag2idx[t] = len(tag2idx)

            if not tags_only:
                word2name[len(word2idx)] = self.padding
                word2idx[self.padding] = len(word2idx)
                char2name[len(char2idx)] = self.padding
                char2idx[self.padding] = len(char2idx)

                word2name[len(word2idx)] = self.unknown
                word2idx[self.unknown] = len(word2idx)
                char2name[len(char2idx)] = self.unknown
                char2idx[self.unknown] = len(char2idx)

            self.encoding = {
                'tag2idx': tag2idx, 
                'tag2name': tag2name,
                'word2idx': word2idx,
                'word2name': word2name,
                'char2idx': char2idx, 
                'char2name': char2name
            }
            self.output_handler.save_json(self.encoding, name='encoding')

    def _prepro_text_file(self, path):
        with path.open(mode='r') as in_f:
            text_in = in_f.read()
        text_prepro = art.get_tokenized_sentences(text_in)
        return text_prepro
    
    def _read_text_file(self, path, bef, aft, read_empty=False):

        def contextWindow(text, bef, aft):
            # 0B, OA --- no change
            if (bef == 0) and (aft == 0):
                return text
            
            # 0B, 1A --- 2 conditions  
            elif (bef == 0) and (aft == 1):
                contxt_txt = []
                for i in range(len(text)):
                    # 0B, 1A
                    if (i >= 0) and (i < len(text)-1):
                        contxt_txt.append(text[i] + text[i+1]) 
                    # 0B, 0A
                    elif i == len(text)-1:
                        contxt_txt.append(text[i]) 
                return contxt_txt
                
            #OB, 2A --- 3 conditions
            elif (bef == 0) and (aft == 2):
                contxt_txt = []
                for i in range(len(text)):
                    #OB, 2A
                    if (i >= 0) and (i <len(text)-2):
                        contxt_txt.append(text[i] + text[i+1]+text[i+2])
                    elif i == (len(text)-2):
                        contxt_txt.append(text[i] + text[i+1])
                    elif i == (len(text)-1):
                        contxt_txt.append(text[i])
                return contxt_txt
            #1B, 0A --- 2 conditions
            elif (bef == 1) and (aft == 0):
                contxt_txt = []
                for i in range(len(text)):
                    #0B, 0A
                    if (i == 0):
                        contxt_txt.append(text[i])
                    elif (i >0):
                        contxt_txt.append(text[i-1]+text[i])
                return contxt_txt
        
            #1B, 1A --- 3 conditions 
            elif ( bef == 1 ) and ( aft == 1):
                contxt_txt = []
                for i in range(len(text)):
                    if i == 0:   # 0B, 1A
                        contxt_txt.append(text[i] + text[i+1])
                    elif (i > 0) and ( i < len(text)-1):  # 1B ,1A
                        contxt_txt.append(text[i-1]+text[i]+text[i+1])
                    elif (i == len(text)-1):  #1B, 0A
                        contxt_txt.append(text[i-1]+text[i])
                return contxt_txt
        
            #1B, 2A --- 4 conditions
            elif (bef ==1 ) and (aft == 2):
                contxt_txt = []
                for i in range(len(text)):
                    if i ==0:
                        contxt_txt.append(text[i]+text[i+1]+text[i+2])                #0B, 2A
                    elif (i > 0) and (i < len(text)-2):
                        contxt_txt.append(text[i-1]+text[i]+text[i+1]+text[i+2])     #1B, 2A
                    elif (i == len(text)-2):
                        contxt_txt.append(text[i-1]+text[i]+text[i+1])               #1B, 1A
                    elif i == (len(text)-1):
                        contxt_txt.append(text[i-1]+text[i])                         #1B,0A
                return contxt_txt
    
            #2B, 0A   -- 3 cases 
            elif( bef ==2) and ( aft == 0):
                contxt_txt = []
                for i in range(len(text)):
                    if i == 0:       
                        contxt_txt.append(text[i])         #0B, 0A
                    elif i == 1:
                        contxt_txt.append(text[i-1]+text[i])  #1B, 0A
                    elif i > 1:
                        contxt_txt.append(text[i-2]+text[i-1]+text[i]) #2B, 0A
                return contxt_txt
        
            #2B, 1A -- 3 cases
            elif ( bef == 2 ) and (aft == 1):
                contxt_txt = []
                for i in range(len(text)):
                    if i ==0: 
                        contxt_txt.append(text[i]+text[i+1])                     #0B, 1A
                    elif i ==1:
                        contxt_txt.append(text[i-1]+text[i]+text[i+1])           #1B, 1A
                    elif (i > 1) and (i < len(text)-1):
                        contxt_txt.append(text[i-2]+text[i-1]+text[i]+text[i+1]) # 2B, 1A
                    elif i == len(text)-1:
                        contxt_txt.append(text[i-2]+text[i-2]+text[i])           #2B, 0A
                return contxt_txt
                
            #2B, 2A --- 5 conditions
            elif (bef == 2) and (aft == 2):        
                contxt_txt = []        
                for i in range(len(text)):    
            
                    if i == 0:    # 0B, 2A
                        contxt_txt.append(text[i] + text[i+1] + text[i+2]) 
                    elif i == 1:  # 1B, 2A
                        contxt_txt.append(text[i-1] + text[i] + text[i+1] + text[i+2]) 
                    elif (i >=2) and (i < len(text)-2):   # 2B , 2A 
                        contxt_txt.append(text[i-2] + text[i-1] + text[i] + text[i+1] + text[i+2]) 
                    elif  (i == len(text)-2):             #2B, 1A
                        contxt_txt.append(text[i-2] + text[i-1] + text[i] + text[i+1])
                    elif i == len(text)-1:                #2B, 0A  
                        contxt_txt.append(text[i-2] + text[i-1] + text[i])            
                return contxt_txt

        text = []
        with path.open(mode='r') as in_f:
            for line in in_f:
                clean_line = line.rstrip()
                if clean_line:
                    text.append(clean_line)
                elif read_empty:
                    text.append([])
        return contextWindow(text, bef, aft)

    def _read_feature_file(self, path):
        features = np.load(str(path), allow_pickle=True)
        if self.feature_dim is None:
            self.feature_dim = features['features'][0].shape[-1]
        return features['features']

    def _read_relation_file(self, path):
        relations = []
        with path.open(mode='r') as in_f:
            for line in in_f:
                if not line.rstrip():
                    relations.append([])
                else:
                    sentence_rels = []
                    rel_strings = line.split(';;')
                    for rel in rel_strings:
                        if rel.rstrip():
                            rel_type, ent_1_str, ent_1_beg, ent_1_num, ent_2_str, ent_2_beg, ent_2_num = rel.split('\t')
                            if self.relation_remapping is not None:
                                rel_type = self.relation_remapping[rel_type]
                            sentence_rels.append({
                                'type': rel_type,
                                'ent1_s': ent_1_str,
                                'ent1_b': ent_1_beg,
                                'ent1_n': ent_1_num,
                                'ent2_s': ent_2_str,
                                'ent2_b': ent_2_beg,
                                'ent2_n': ent_2_num
                            })
                    relations.append(sentence_rels)
        return relations

    def _collate_fn(self, batch):
        characters = [x['characters'] for x in batch]
        ids = [x['ids'] for x in batch]
        tags = [x['tags'] for x in batch]
        features = [x['features'] for x in batch]

        max_word_length_batch = 0
        if self.max_word_length > 0:
            max_word_length_batch = self.max_word_length
        else:
            max_word_length_batch = 0
            for sent in characters:
                if sent.size()[-1] > max_word_length_batch:
                    max_word_length_batch = sent.size()[-1]

        length = torch.tensor([x.size() for x in ids])
        padded_ids = pad_sequence(ids, batch_first=True, padding_value=self.encoding['word2idx'][self.padding])
        padded_tags = pad_sequence(tags, batch_first=True, padding_value=self.encoding['tag2idx']['O'])
        
        max_sent_length_batch = 0
        if self.max_sent_length > 0:
            padded_ids = F.pad(padded_ids, (0, self.max_sent_length-padded_ids.shape[-1]), "constant", self.encoding['word2idx'][self.padding])
            padded_tags = F.pad(padded_tags, (0, self.max_sent_length-padded_tags.shape[-1]), "constant", self.encoding['tag2idx']['O'])
            characters = [F.pad(x, (0,(max_word_length_batch-x.size()[-1]),0,(self.max_sent_length-x.size()[0])), "constant", self.encoding['char2idx'][self.padding]).unsqueeze(0) for x in characters]
        else:
            characters = [F.pad(x, (0,(max_word_length_batch-x.size()[-1]),0,(padded_ids.size()[-1]-x.size()[0])), "constant", self.encoding['char2idx'][self.padding]).unsqueeze(0) for x in characters]

        if features[0] is not None:
            padded_features = pad_sequence(features, batch_first=True)
            if self.max_sent_length > 0:
                padded_features = F.pad(padded_features, (0, 0, 0, self.max_sent_length-padded_features.shape[-2]), "constant", 0)        
        else:
            padded_features = None
            
        characters = torch.cat(characters, dim=0)
        return {
            'chars': characters, 
            'ids': padded_ids, 
            'tags': padded_tags, 
            'lengths': length, 
            'features': padded_features
        }

    def _prepare_prediction_data(self, sentences, features, tags=[]):
        character_ids = []
        input_ids = []
        tag_ids = []
        feat_ids = []
        for sentence, feat, tag in zip_longest(sentences, features, tags, fillvalue=[]):
            # TODO how to handle unknowns?
            character_sentence = [[self.encoding['char2idx'][i] if i in self.encoding['char2idx'] else self.encoding['char2idx'][self.unknown] for i in w] for w in sentence]
            tokenized_sentence = [self.encoding['word2idx'][w] if w in self.encoding['word2idx'] else self.encoding['word2idx'][self.unknown] for w in sentence]

            character_ids.append(character_sentence)
            input_ids.append(tokenized_sentence)
            tag_ids.append(tag)
            feat_ids.append(feat) 

        return character_ids, input_ids, tag_ids, feat_ids

    def _prepare_training_data(self, sentences, tags, features, keep_prob=1.0):
        character_ids = []
        input_ids = []
        tags_ids = []
        feat_ids = []
        for sentence, tag, feat in zip_longest(sentences, tags, features, fillvalue=[]):
            labels = tag.split()
            
            if any([not t.startswith('O') for t in labels]) or math.isclose(keep_prob, 1.0, rel_tol=1e-5) or random.random() < keep_prob: 
                words = sentence.split()
                labels = [self.encoding['tag2idx'][self._adjust_tag(t)] for t in labels]
                character_sentence = [[self.encoding['char2idx'][i] for i in j] for j in words]
                tokenized_sentence = [self.encoding['word2idx'][w] for w in words]
                
                character_ids.append(character_sentence)
                input_ids.append(tokenized_sentence)
                tags_ids.append(labels)     
                feat_ids.append(feat)  

        return character_ids, input_ids, tags_ids, feat_ids

    def _prepare_bert_prediction_data(self, sentences, tags=[]):
        input_ids = []
        if not self.multi_task_mapping:
            tags_ids = []
        else:
            tags_ids = {}
            for k in self.encoding['tag2idx'].keys():
                tags_ids[k] = []
        attention_masks = []
        length = []
        for sentence, tag in zip_longest(sentences, tags, fillvalue=[]):            
            tokenized_sentence = []
            for word in sentence:
                tokenized_word = self.tokenizer.tokenize(word)
                tokenized_sentence.extend(tokenized_word)
        
            inputs = self.tokenizer.encode_plus(tokenized_sentence, add_special_tokens=True, return_attention_mask=True)

            input_ids.append(inputs["input_ids"])
            attention_masks.append(inputs['attention_mask'])
            length.append(min(len(inputs['input_ids']), BERT_MAX_LENGTH))
            if not self.multi_task_mapping:
                tags_ids.append(tag)
            else:
                for mapping_key in self.encoding['tag2idx'].keys():
                    tags_ids[mapping_key].append(tag)
        
        return input_ids, tags_ids, attention_masks, length

    def _prepare_bert_training_data(self, sentences, tags, keep_prob):
        input_ids = []
        if not self.multi_task_mapping:
            tags_ids = []
        else:
            tags_ids = {}
            for k in self.tag_remapping.keys():
                tags_ids[k] = []
        attention_masks = []
        length = []
        for sentence, tag in zip(sentences, tags):
            labels = tag.split()
            
            if any([not t.startswith('O') for t in labels]) or math.isclose(keep_prob, 1.0, rel_tol=1e-5) or random.random() < keep_prob: 
                tokenized_sentence = []
                tokenized_labels = []
                words = sentence.split()
                for word, label in zip(words, labels):
                    tokenized_word = self.tokenizer.tokenize(word)
                    n_subwords = len(tokenized_word)
                    tokenized_sentence.extend(tokenized_word)
                    if not label.startswith('B-'):
                        tokenized_labels.extend([label] * n_subwords)
                    else:
                        tokenized_labels.extend([label])
                        tokenized_labels.extend([label.replace('B-', 'I-')] * (n_subwords-1))
            
                inputs = self.tokenizer.encode_plus(tokenized_sentence, add_special_tokens=True, return_attention_mask=True)
                tokenized_labels = ["O"] + tokenized_labels + ["O"]

                input_ids.append(inputs["input_ids"])
                attention_masks.append(inputs['attention_mask'])
                length.append(min(len(inputs["input_ids"]), BERT_MAX_LENGTH))

                if not self.multi_task_mapping:
                    label_ids = [self.encoding['tag2idx'][self._adjust_tag(t)] for t in tokenized_labels]
                    tags_ids.append(label_ids)
                else:
                    for mapping_key in self.tag_remapping.keys():
                        label_ids = [self.encoding['tag2idx'][mapping_key][self._adjust_tag(t, mapping_key)] for t in tokenized_labels]
                        tags_ids[mapping_key].append(label_ids)
        
        return input_ids, tags_ids, attention_masks, length

    def load_input(self, bef, aft):
        for dataset, dataset_setup in self.data_config['sets'].items():
            for sub_dataset in dataset_setup:

                sub_dataset['sentences'] = []
                sub_dataset['tags'] = []
                sub_dataset['features'] = []
                sub_dataset['relations'] = []
                
                for file_config in sub_dataset['all_files']:
                    if self.data_file_extension:
                        sub_dataset['sentences'].extend(self._read_text_file(file_config[self.data_file_extension], bef, aft))
                    if self.label_file_extension:
                        sub_dataset['tags'].extend(self._read_text_file(file_config[self.label_file_extension], bef , aft))
                    if self.feature_file_extension:
                        sub_dataset['features'].extend(self._read_feature_file(file_config[self.feature_file_extension]))
                    if self.relation_file_extension:
                        sub_dataset['relations'].extend(self._read_relation_file(file_config[self.relation_file_extension]))
                
    def _pad_to_length(self, sequences, length=512, fill_value=0, dtype=torch.long):
        out_seqs = torch.full((len(sequences), length), fill_value, dtype=dtype)
        for idx, ids in enumerate(sequences):
            ids_to_insert = ids[:length]
            out_seqs[idx, :len(ids_to_insert), ...] = torch.tensor(ids_to_insert)
        return out_seqs

    def data_loaders(self):
        for dataset, dataset_setup in self.data_config['sets'].items():
            for idx, sub_dataset in enumerate(dataset_setup):
                if self.tokenizer is None:
                    characters, ids, tags, features = self._prepare_training_data(sub_dataset['sentences'], sub_dataset['tags'], sub_dataset['features'], sub_dataset['keep_neg_sample_prob'])
                    input_data = LSTMDataset(characters, ids, tags, features, self.encoding['char2idx'], self.padding, self.max_word_length, self.max_sent_length)
                    sampler = RandomSampler(input_data)
                    sub_dataset['dataloader'] = DataLoader(input_data, sampler=sampler, batch_size=self.batch_size, collate_fn=self._collate_fn)
                else: 
                    input_ids, tag_ids, attention_masks, lengths = self._prepare_bert_training_data(sub_dataset['sentences'], sub_dataset['tags'], sub_dataset['keep_neg_sample_prob'])
                    input_ids = self._pad_to_length(input_ids, length=BERT_MAX_LENGTH, fill_value=self.special_toks['pad_tok'], dtype=torch.long)
                    attention_masks = self._pad_to_length(attention_masks, length=BERT_MAX_LENGTH, fill_value=0, dtype=torch.long)
                    if not self.multi_task_mapping:
                        tag_ids = self._pad_to_length(tag_ids, length=BERT_MAX_LENGTH, fill_value=self.encoding['tag2idx']['O'], dtype=torch.long)
                        input_data = BERTDataset(input_ids, tag_ids, attention_masks)
                    else:                            
                        for k in tag_ids.keys():
                            tag_ids[k] = self._pad_to_length(tag_ids[k], length=BERT_MAX_LENGTH, fill_value=self.encoding['tag2idx'][k]['O'], dtype=torch.long)
                        input_data = BERTMultiDataset(input_ids, tag_ids, attention_masks, lengths)
                    
                    sampler = RandomSampler(input_data)
                    sub_dataset['dataloader'] = DataLoader(input_data, sampler=sampler, batch_size=self.batch_size)

    def word_embedding(self, emb_conf):
        if self.checkpoint is not None and self.checkpoint['model']:
            print("Using a pre-trained model --- word embedding is loaded as part of the model")
            embedding_weights = torch.zeros(len(self.encoding['word2idx']), emb_conf['dim'])
        else:
            print("Loading word embedding: {}".format(emb_conf['file']))
            pretrained_word_vectors = KeyedVectors.load_word2vec_format(emb_conf['file'], binary=True)
            if emb_conf['zero_init']:
                embedding_weights = torch.zeros(len(self.encoding['word2idx']), emb_conf['dim'])
            else:
                embedding_weights = torch.randn(len(self.encoding['word2idx']), emb_conf['dim'])
            unknown_word_count = 0
            for word, idx in self.encoding['word2idx'].items():
                try:
                    embedding_weights[idx] = torch.tensor(pretrained_word_vectors[word].copy())
                except KeyError:
                    unknown_word_count += 1
            print("{}/{} word from the dataset do no exist in the word embedding".format(unknown_word_count, len(self.encoding['word2idx'])))
        return embedding_weights
