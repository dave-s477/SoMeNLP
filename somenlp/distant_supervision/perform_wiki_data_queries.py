import re
import json
import requests
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

from urllib.parse import quote_plus

WIKIDATA_ID = re.compile(r'Q\d{5,}')

def parse_query(plain_query, lang, default_address='https://query.wikidata.org/bigdata/namespace/wdq/sparql?query='):
    """Encode Wikidata query to pass it as an URL

    Args:
        plain_query (str): plain text query
        lang (str): wikidata language identifier for the query
        default_address (str, optional): url for wikidata sparql interface. Defaults to 'https://query.wikidata.org/bigdata/namespace/wdq/sparql?query='.

    Returns:
        str: encoded query
    """
    plain_query = re.sub("'en'", '"' + lang + '"', plain_query)
    encoded_query = quote_plus(plain_query, safe=r"()\{\}")
    encoded_query = re.sub(r"\+", "%20", encoded_query)
    complete_url = '{}{}'.format(default_address, encoded_query)
    return complete_url

def execute_query(query_url):
    """Perform wikidata query. Might not be handled if no user agent is provided.

    Args:
        query_url (str): url query to execute

    Returns:
        str: json formatted response text
    """
    #user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
    user_agent = ''
    headers = {
        'Accept': 'json',
        'User-Agent': user_agent
    }
    response = requests.get(query_url, headers=headers)
    if response.status_code != 200:
        print("Received HTTP-Statuscode " + str(response.status_code))
        if response.status_code == 400:
            print("Error is probably in the query syntax.")
            sys.exit(1)
        elif response.status_code == 429:
            print("Too many requests were sent: ")
            print("Sleeping: " + int(response.headers["Retry-After"]))
            time.sleep(int(response.headers["Retry-After"]))
            print("Continuing with query")
            response = requests.get(query_url, headers=headers)
            if response.status_code != 200:
                raise(RuntimeError("Got another error while querying wikidata {}\nShutting down..".format(response.status_code)))
            else:
                return response.text
    else:
        return response.text

def query_wikidata(query_config):
    """Perform a series of wikidata queries based on a given configuration

    Args:
        query_config (dictionary): configuration of queries saved as json

    Returns:
        dictionary: merged responses for given queries
    """
    results = {}
    with open(query_config) as qc:
        queries = json.load(qc)
    for query_name, query_data in queries.items():
        print("Performing wikidata queries for {}".format(query_name))
        target_main_names = '{}_main_name'.format(query_data['target'])
        target_alt_names = '{}_alt_name'.format(query_data['target'])
        if target_main_names not in results.keys():
            results[target_main_names] = set()
            results[target_alt_names] = set()
        for lang in query_data['languages']:
            count = 0
            query_string = parse_query(query_data["query_string"], lang)
            query_response = execute_query(query_string)
            soup = BeautifulSoup(query_response, 'lxml')
            for res in soup.findAll('result'):
                for bind in res.findAll('binding'):
                    count += 1
                    if bind['name'] == 'itemLabel':
                        results[target_main_names].update([bind.text.rstrip().lstrip()])
                    elif bind['name'] == 'abbreviation':
                        results[target_alt_names].update([bind.text.rstrip().lstrip()])
            print("Processed {} entries for lang {}".format(count, lang))
    return results
