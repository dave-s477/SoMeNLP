import json

from articlenizer import articlenizer as art

def load_dicts(locations):
    """Load and merge information from json dictionaries

    Args:
        locations (list of Posix paths): locations to json files

    Returns:
        dictionary: dictionary merged from json files 
    """
    results = {}
    for f in locations:
        name = f.name.split('.dic')[0]
        results[name] = []
        with f.open(mode='r', errors='ignore') as in_f:
            for line in in_f:
                if line.rstrip():
                    results[name].append(line.rstrip())
    return results

def normalize_entries(entries, max_length):
    """Shorten entries in list based on number of tokens after tokenization

    Args:
        entries (list): list of candidates
        max_length (int): max allowed token length

    Returns:
        list: list with filtered long entries
    """
    entry_list = []
    for entry in entries:
        tokens = art.tokenize_text(entry)
        if len(tokens) <= max_length:
            entry_list.append(' '.join(tokens))
    return entry_list

def merge_results(out_file, max_token_length, inputs):
    """Merge and write a distant supervision ditionary from filtered dictionaries

    Args:
        out_file (Posix path): output location
        max_token_length (int): max allowed token length
        inputs (list of dictionaries): individual dicts to be merged
    """
    distant_supervision_dictionary = {}
    for i in inputs:
        if i is not None:
            for k, v in i.items():
                distant_supervision_dictionary[k] = normalize_entries(v, max_token_length)
    with out_file.open(mode='w') as json_out:
        json.dump(distant_supervision_dictionary, json_out, indent=4)
