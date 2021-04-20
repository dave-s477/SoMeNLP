import numpy as np

def distant_supervision_by_dict(candidate, dictionary, mapping):
    result = np.zeros(len(mapping), dtype=np.bool)
    span_to_examine = candidate.base_span
    values = [mapping[x] for x in dictionary[span_to_examine]] if span_to_examine in dictionary.keys() else []
    if values:
        for val in values:
            result[val] = True
    return result