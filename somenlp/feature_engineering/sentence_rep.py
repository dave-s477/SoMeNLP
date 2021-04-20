import string
import nltk
import re
import unicodedata
import numpy as np

from nltk.corpus import wordnet

from . import word_rules

lemmatizer = nltk.stem.WordNetLemmatizer()

URL = re.compile(r"^(https?\:\/\/[a-zA-Z0-9\-\.]+[\w\/\._\-\:~\?=#%]*[\w\/_\-\:~\?=#%]|ftp\:\/\/[a-zA-Z0-9\-\.]+[\w\/\._\-\:~\?=#%]*[\w\/_\-\:~\?=#%]|www\.[a-zA-Z0-9\-\.]+[\w\/\._\-\:~\?=#%]*|[a-zA-Z0-9\-\.]+\.(org|edu)/[\w\/_\-\:~\?=#%]*)$") 
CITATION = re.compile(r'^\[[0-9\-,\?]+\]$')

FLOAT_NUM = re.compile(r'^\d+\.\d*$')
FLOAT_NON_LEADING = re.compile(r'^\.\d+$')
VERSION_LIKE_NUM = re.compile(r'^(\d+\.){2}\w*$')
LONG_VERSION_LIKE = re.compile(r'^(\d+\.){3,8}\w*$')
LONG_NUM = re.compile(r'^(\d{1,3}\,){1,8}\d{3}$')

HEADWORDS = ['software', 'package', 'program', 'tool', 'toolbox', 'web', 'service', 'spreadsheet', 'database', 'registry', 'data', 'model', 'algorithm', 'kit', 'standard', 'method', 'procedure']

def get_wordnet_pos(tag):
    """Map POS tag to first character lemmatize() accepts"""
    s_tag = tag[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(s_tag, wordnet.NOUN)

class SentenceRepresentation:
    def __init__(self, input_string):
        self.features = {}

        # Basic words
        self.tokens = input_string.split()
        self.tokens_lower = [x.lower() for x in self.tokens]
        pos_tags = nltk.pos_tag(self.tokens)
        self.lemmas = []
        for w, t in pos_tags:
            wntag = get_wordnet_pos(t)
            self.lemmas.append(lemmatizer.lemmatize(w, wntag))
        self.length = len(self.tokens)
        self.pos_tags = [x[1] for x in pos_tags]

        # additional information
        self.features['token_length'] = [len(x) for x in self.tokens]
        self.features['punct'] = [x in string.punctuation for x in self.tokens]
        self.features['math_chars'] = [unicodedata.category(x) == 'Sm' if len(x) == 1 else False for x in self.tokens] 
        self.features['hypen'] = [x == '-' for x in self.tokens]
        self.features['slash'] = [x == '/' for x in self.tokens]
        self.features['bracket_open'] = [x == '(' for x in self.tokens]
        self.features['bracket_close'] = [x == ')' for x in self.tokens]

        # digit information
        self.features['digit'] = [x.isdigit() for x in self.tokens]
        self.features['float_num'] = [bool(FLOAT_NUM.match(x)) for x in self.tokens]
        self.features['float_non_leading'] = [bool(FLOAT_NON_LEADING.match(x)) for x in self.tokens]
        self.features['version_like_num'] = [bool(VERSION_LIKE_NUM.match(x)) for x in self.tokens]
        self.features['long_version_like'] = [bool(LONG_VERSION_LIKE.match(x)) for x in self.tokens]
        self.features['long_num'] = [bool(LONG_NUM.match(x)) for x in self.tokens]  

        # further token based information
        self.features['citation'] = [bool(CITATION.match(x)) for x in self.tokens]
        self.features['url'] = [bool(URL.match(x)) for x in self.lemmas]

        # casing info
        self.features['upper'] = [word_rules.upper_cased(x) for x in self.tokens]
        self.features['first_char_upper'] = [word_rules.first_char_upper(x) for x in self.tokens]
        self.features['mixed_case'] = [word_rules.mixed_case(x) for x in self.tokens]
        self.features['lower_case'] = [word_rules.lower_case(x) for x in self.tokens]

        # individual head words
        for headword in HEADWORDS:
            self.features[headword] = [x == headword for x in self.lemmas]  

    def get_features(self):
        matrix = []
        for _, v in self.features.items():
            matrix.append(v)
        return np.array(matrix)
        
    def get_left_tokens(self, idx, size, style='plain'):
        if style == 'plain':
            return self.tokens[max(idx-size, 0):idx]
        if style == 'lower':
            return self.tokens_lower[max(idx-size, 0):idx]
        elif style == 'lemma':
            return self.lemmas[max(idx-size, 0):idx]
        else:
            raise(RuntimeError("Style '{}' is not defined".format(style)))

    def get_right_tokens(self, idx, size, style='plain'):
        if style == 'plain':
            return self.tokens[idx:idx+size]
        if style == 'lower':
            return self.tokens_lower[idx:idx+size]
        elif style == 'lemma':
            return self.lemmas[idx:idx+size]
        else:
            raise(RuntimeError("Style '{}' is not defined".format(style)))

    def get_candidate(self, start_idx, end_idx):
        return self.tokens[start_idx:end_idx], self.lemmas[start_idx:end_idx]

class Candidate:
    def __init__(self, sentence, start_idx, end_idx):
        self.tokens, self.lemmas = sentence.get_candidate(start_idx, end_idx)
        self.sentence = sentence
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.base_span = ' '.join(self.tokens)
