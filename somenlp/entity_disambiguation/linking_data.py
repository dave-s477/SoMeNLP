from hashlib import new
import json
import sys
import random
import re
import numpy as np

from articlenizer import articlenizer
from nltk.corpus import stopwords

random.seed(42)
STOPS = stopwords.words('english')

def normalize(s):
    norm_s = re.sub('[^0-9a-zA-Z]+', ' ', s.casefold()).rstrip('0123456789 ,.').lstrip(' ')
    norm_s = ' '.join([w for w in norm_s.split() if w not in STOPS])
    if not norm_s:
        norm_s = s
    return norm_s

def remove_spaces(s):
    replace_regex = re.compile(r'\s(?P<to_keep>[\+\-#™_/\d]+)\s?')
    return replace_regex.sub(r'\g<to_keep>', s)

class LinkingData:
    def __init__(self, config):
        self.config = config
        self.read_data_files(config)

    def read_data_files(self, config):
        self.train = {}
        for entry in config['train']:
            with open(entry['file'], 'r') as data_f:
                self.train[entry['name']] = json.load(data_f)
                for idx, entry in enumerate(self.train[entry['name']]):
                    entry['init_id'] = idx

        self.test = {}
        for entry in config['test']:
            with open(entry['file'], 'r') as data_f:
                self.test[entry['name']] = json.load(data_f)
                for idx, entry in enumerate(self.test[entry['name']]):
                    entry['init_id'] = idx
        
        if config['gold']:
            with open(config['gold'], 'r') as gold_f:
                self.gold = json.load(gold_f)

        if self.config['data_augmentation']:
            self.all = []
            self.all_names = []
            self.all_augmented_names = []
            self.short_samples = []
            self.long_samples = []
            with open(config['all'], 'r') as data_f:
                all = json.load(data_f)
            for sample in all:
                self.all_names.append(sample['mention'].lower())
                tokens = sample['mention'].split()
                sample['tokens'] = tokens
                self.all.append(sample)
                if len(tokens) <= 1:
                    self.short_samples.append(sample)
                else:
                    self.long_samples.append(sample)
            self._get_augmented_samples()

            for s in self.all:
                for idx, g in enumerate(self.gold):
                    if s['paper_id'] == g['paper_id'] and s['mention'] == ' '.join(articlenizer.get_tokenized_sentences(g['mention'])[0]):
                        s['gold_id'] = idx
                        break

            for s in self.augmented_samples:
                self.all.append(s)   
            random.shuffle(self.all) 

            self.link_lookup_table = np.zeros((len(self.all), len(self.all)), dtype=bool)
            for idx, s in enumerate(self.all):
                s['origin_id'] = idx
                s['string'] = remove_spaces(s['mention'])
                s['norm'] = normalize(s['mention'])
                for idx_2, s_2 in enumerate(self.all[idx+1:]):
                    if 'gold_id' in s and 'gold_id' in s_2 and self.gold[s['gold_id']]['link'] == self.gold[s_2['gold_id']]['link']:
                        self.link_lookup_table[idx][idx+1+idx_2] = True
                        self.link_lookup_table[idx+1+idx_2][idx] = True

            self.all_names = None
            self.all_augmented_names = None
            self.short_samples = None
            self.long_samples = None

    def _get_augmented_samples(self):
        num_augmentation_samples = 0
        for _, v in self.train.items(): 
            num_augmentation_samples += len(v)
        print(num_augmentation_samples)
        num_augmentation_samples *= self.config['num_augmentation_samples']
        self.augmented_samples = []
        for _ in range(int(num_augmentation_samples/2)):
            # Augment short sample
            s_1 = self.short_samples[random.randint(0, len(self.short_samples)-1)]
            s_2 = self.short_samples[random.randint(0, len(self.short_samples)-1)]
            self._recombine(s_1, s_2)

            # Augment long sample
            s_1 = self.long_samples[random.randint(0, len(self.long_samples)-1)]
            s_2 = self.long_samples[random.randint(0, len(self.long_samples)-1)]
            self._recombine(s_1, s_2)

    def _recombine_words(self, s1, s2):
        split_idx_1 = random.randint(1, len(s1)-1) if len(s1) > 1 else random.randint(0,1)
        split_idx_2 = random.randint(1, len(s2)-1) if len(s2) > 1 else random.randint(0,1)
        if isinstance(s1, str) and isinstance(s2, str):
            s1_new = s1[:split_idx_1] + s2[split_idx_2:]
            s2_new = s2[:split_idx_2] + s1[split_idx_1:]
        elif isinstance(s1, list) and isinstance(s2, list):
            s1_new = ' '.join(s1[:split_idx_1] + s2[split_idx_2:])
            s2_new = ' '.join(s2[:split_idx_2] + s1[split_idx_1:])
        elif isinstance(s1, str) and isinstance(s2, list):
            s1_new = s1[:split_idx_1] + ' ' + ' '.join(s2[split_idx_2:])
            s2_new = ' '.join(s2[:split_idx_2]) + ' ' + s1[split_idx_1:]
        elif isinstance(s1, list) and isinstance(s2, str):
            s1_new = ' '.join(s1[:split_idx_1]) + ' ' + s2[split_idx_2:]
            s2_new = s2[:split_idx_2] + ' ' + ' '.join(s1[split_idx_1:])
        return s1_new, s2_new
            
    def _get_add_info(self, r):
        i = {}
        for v in r:
            tokens = v['string'].split()
            i[v['type']] = tokens if len(tokens) > 1 else v['string']
        return i
    
    def _recombine(self, s_1, s_2):
        if s_1['mention'].lower() == s_2['mention'].lower():
            return []
        new_strings = self._recombine_words(s_1['mention'], s_2['mention'])
        add_info_1 = self._get_add_info(s_1['relations'])
        add_info_2 = self._get_add_info(s_2['relations'])
        for k, v in add_info_1.items():
            if k in add_info_2 and v == add_info_2[k]:
                return []
        new_developers = self._recombine_words(add_info_1['Developer_of'], add_info_2['Developer_of']) if 'Developer_of' in add_info_1 and 'Developer_of' in add_info_2 else None
        new_url = self._recombine_words(add_info_1['URL_of'], add_info_2['URL_of']) if 'URL_of' in add_info_1 and 'URL_of' in add_info_2 else None
        new_version = self._recombine_words(add_info_1['Version_of'], add_info_2['Version_of']) if 'Version_of' in add_info_1 and 'Version_of' in add_info_2 else None
        for idx, n_s in enumerate(new_strings):
            if not n_s.lower() in self.all_names and not n_s.lower() in self.all_augmented_names:
                new_sample = {
                    'mention': n_s,
                    'relations': []
                }
                if new_developers is not None:
                    new_sample['relations'].append({
                        'type': 'Developer_of',
                        'string': new_developers[idx]
                    })
                if new_url is not None:
                    new_sample['relations'].append({
                        'type': 'URL_of',
                        'string': new_url[idx]
                    })
                if new_version is not None:
                    new_sample['relations'].append({
                        'type': 'Version_of',
                        'string': new_version[idx]
                    })
                self.augmented_samples.append(new_sample)
                self.all_augmented_names.append(n_s)
