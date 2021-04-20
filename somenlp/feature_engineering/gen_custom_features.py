import json
import numpy as np

from  multiprocessing import Pool
from functools import partial

from . import candidate_rules
from . import distant_supervision_rules
from .sentence_rep import SentenceRepresentation, Candidate

FUNCTION_NAMES = [
    'pan_top_1', 'pan_top_2', 'pan_top_3', 'pan_top_4', 'pan_top_5', 'pan_top_6', 'pan_top_7', 'pan_top_8', 'text_is_in_brackets', 'developer'] 
fcts = [getattr(candidate_rules, n) for n in FUNCTION_NAMES]

def encode_sentence(text, max_length, distant_supervision_dict, key_mapping, pos_tag_encoding):
    sentence = SentenceRepresentation(text)

    lengths = sentence.length
    pos_tags = np.expand_dims(np.array([pos_tag_encoding[x] for x in sentence.pos_tags]), 0)
    features = sentence.get_features()

    sentence_counts = np.zeros((len(FUNCTION_NAMES), sentence.length), np.int32)
    distant_supervision_counts = np.zeros((len(key_mapping), sentence.length), np.int32)
    for cand_length in range(1, max_length+1):
        for cand_beg in range((sentence.length - cand_length) + 1):
            cand_end = cand_beg + cand_length
            candidate = Candidate(sentence, cand_beg, cand_end)
            for fct_idx, fct in enumerate(fcts):                
                match_res = fct(candidate)
                if match_res == 1:
                    for pos in range(candidate.start_idx, candidate.end_idx):
                        sentence_counts[fct_idx][pos] += 1

            distant_supervision_result = distant_supervision_rules.distant_supervision_by_dict(candidate, distant_supervision_dict, key_mapping)
            for pos in range(candidate.start_idx, candidate.end_idx):
                distant_supervision_counts[:,pos] += distant_supervision_result

    all_feats = np.concatenate((pos_tags, features, sentence_counts, distant_supervision_counts))
    return np.transpose(all_feats)

def calculate_features(input_files, max_length, distant_supervision_dict, key_mapping, pos_tag_encoding):
    sentence_feature_encodings = []
    with input_files[0].open(mode='r') as in_text:
        for line in in_text:
            sentence_feature_encodings.append(encode_sentence(line, max_length, distant_supervision_dict, key_mapping, pos_tag_encoding)) 
    np.savez_compressed('{}.npz'.format(str(input_files[1])), features=sentence_feature_encodings)

def update_words(all_words, in_list, name):
    for word in in_list:
        if word not in all_words.keys():
            all_words[word] = []
        if name not in all_words[word]:
            all_words[word].append(name)

def load_distant_supervision(loc):
    with loc.open(mode='r') as in_f:
        distant = json.load(in_f)
    all_distsup_words = {}
    key_mapping = {}
    for k, v in distant.items():
        update_words(all_distsup_words, v, k)
        key_mapping[k] = len(key_mapping)
    return all_distsup_words, key_mapping

def load_pos_tag_encoding(loc):
    with loc.open(mode='r') as in_f:
        encoding = json.load(in_f)
    return encoding

def calculate_features_parallel(files, max_length, distant_supervision_dictionary_location, pos_tag_encoding_location, n_cores=8):
    distant_supervision_dictionary, distant_supervision_key_mapping = load_distant_supervision(distant_supervision_dictionary_location)
    pos_tag_encoding = load_pos_tag_encoding(pos_tag_encoding_location)
    fct_to_execute = partial(calculate_features, max_length=max_length, distant_supervision_dict=distant_supervision_dictionary, key_mapping=distant_supervision_key_mapping, pos_tag_encoding=pos_tag_encoding)
    with Pool(n_cores) as p:
        p.map(fct_to_execute, files)
        