import re

from itertools import combinations
from articlenizer import CUSTOM_STOPWORDS

def acronym(tokens_in):
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

def is_substring(e1, e2):
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

def get_context_word(sentence, position):
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

def one_hot_encoding(dictionary, name, ent_type, encoding):
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

def get_entities_inbetween(tags, ind1, ind2):
    """Get number of tagged elements inbetween two indicies

    Args:
        tags (list): sequence of tags
        ind1 (int): index
        ind2 (int): index

    Returns:
        int: number of tagged entities
    """
    sub_tags = tags[ind1:ind2].split()
    return sum([1 if x.startswith('B-') else 0 for x in sub_tags])

def get_features(pair, sentence, tags, main_entity_count, entity_encoding):
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

    ent_0_left_context = get_context_word(sentence, pair[0]['beg'])
    ent_1_left_context = get_context_word(sentence, pair[1]['beg'])
    ent_0_tokens = pair[0]['string'].split()
    ent_1_tokens = pair[1]['string'].split()
    ent_0_acronym = acronym(ent_0_tokens)
    ent_1_acronym = acronym(ent_1_tokens)

    entities_in_between = get_entities_inbetween(tags, smaller['end'], larger['beg'])
    
    features = {
        'entity_distance_abs': len(distance_string), 
        'entity_distance_rel': len(distance_string) / len(sentence), 
        'entity_distance_tok': len(tokens_distance_string), 
        'entity_distance_tok_rel': len(tokens_distance_string) / len(sentence.split()), 
        'entity_order': pair[1]['beg'] > pair[0]['beg'], 
        'entity_0_char_length': pair[0]['end'] - pair[0]['beg'],
        'entity_0_token_length': len(ent_0_tokens),
        'entity_0_left_context_for': ent_0_left_context == 'for', 
        'entity_1_char_length': pair[1]['end'] - pair[1]['beg'],
        'entity_1_token_length': len(ent_1_tokens),
        'entity_1_left_context_for': ent_1_left_context == 'for', 
        'num_main_entities': main_entity_count,
        'entity_0_substring_of_entity_1': is_substring(pair[0]['string'], pair[1]['string']),
        'acronym_entity_0_substring_of_entity_1': is_substring(ent_0_acronym, pair[1]['string']), 
        'acronym_entity_0_substring_of_acronym_entity_1': is_substring(ent_0_acronym, ent_1_acronym),
        'entity_1_substring_of_entity_0': is_substring(pair[1]['string'], pair[0]['string']),
        'acronym_entity_1_substring_of_entity_0': is_substring(ent_1_acronym, pair[0]['string']), 
        'acronym_entity_1_substring_of_acronym_entity_0': is_substring(ent_1_acronym, ent_0_acronym)
    }

    one_hot_encoding(features, 'entity_0_type', pair[0]['type'], entity_encoding)
    one_hot_encoding(features, 'entity_1_type', pair[1]['type'], entity_encoding)
    return features


def get_sentence_relations_and_features(sentence, tags, entities, relations, entity_encoding, main_entities=['software']):
    """Span all potential relations in sentence and calculate its respective features

    Args:
        sentence (list): sentence as token list
        tags (list): IOB2 labels assigned to sentence
        entities (dict): entities in sentence
        relations (dict): true relations in sentence
        entity_encoding (dictionary): numeric encoding of entities
        main_entities (list, optional): entity tags identifying main entities. Defaults to ['software'].

    Returns:
        list: relations in sentence including features and labels
    """
    sentence_feature_list = []
    
    main_entity_count = 0
    for ent in entities:
        if ent['type'] in main_entities:
            main_entity_count += 1

    entity_combinations = combinations(entities, 2)
    entity_pairs = []
    for comb in entity_combinations:
        entity_pairs.extend([list(comb), list(reversed(comb))])

    assigned_count = 0
    for idx, pair in enumerate(entity_pairs):
        features = get_features(pair, sentence, tags, main_entity_count, entity_encoding)
        features['label'] = 'none'
        for relation in relations:
            if int(relation['ent1_b']) == int(pair[0]['beg']) and int(relation['ent2_b']) == int(pair[1]['beg']):
                features['label'] = relation['type']
                assigned_count += 1
                break
        sentence_feature_list.append(features)

    return sentence_feature_list
