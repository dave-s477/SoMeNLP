import json
import wget

from pathlib import Path

def load_wiktionary(download_location='/tmp/', default_address='https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.json'):
    """Download a pre-processed wiktionary dump to generate an english dictionary for distant supervision. 

    Args:
        download_location (str, optional): where to write the download. Defaults to '/tmp/'.
        default_address (str, optional): url from where to download the dump. Defaults to 'https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.json'.

    Returns:
        dictionary: list of english words from Wiktionary
    """
    print("Loading English dictionary")
    dict_location = '{}/{}'.format(download_location.rstrip('/'), 'wiktionary_dict.json')
    wget.download(default_address, out=dict_location)
    results = {}
    with open(dict_location, 'r') as w_file:
        for line in w_file:
            entry = json.loads(line)
            if entry['pos'] not in results.keys():
                results[entry['pos']] = set()
            results[entry['pos']].update([entry['word']])

    return results
