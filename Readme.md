# SoMeNLP

SoMeNLP provides functionality for performing information extraction for software mentions in scientific articles. 
Implemented are: Named Entity Recognition, Relation Extraction and Entity Disambiguation. 

Up to now it has been applied on the SoMeSci dataset (available from [zenodo](https://zenodo.org/record/4701764) or [github](https://github.com/dave-s477/SoMeSci)). You can also have a look at the website including an interactive SPARQL Endpoint: https://data.gesis.org/somesci.


Below is a short tutorial for training a NLP pipeline on SoMeSci (or on `data/minimal_example` that is included in the repository)

## Installing

SoMeNLP is structured as a Python package containing code and command line scripts. 
It is absolutely crucial that Python >= 3.6 is used because the insertion order of dictionaries has to be retained.

The package can be installed by: 
```shell
git clone https://github.com/dave-s477/SoMeNLP
cd SoMeNLP
pip install .
```
or for an editable installation
```shell
pip install -e .
```

There is a long list of dependencies that come with the install:
[gensim](https://pypi.org/project/gensim/), [pytorch](https://pypi.org/project/torch/), [tensorboard](https://pypi.org/project/tensorboard/), [articlenizer](https://github.com/dave-s477/articlenizer), [pandas](https://pypi.org/project/pandas/), [numpy](https://pypi.org/project/numpy/), [beautifulsoup](https://pypi.org/project/beautifulsoup4/), [wiktextract](https://pypi.org/project/wiktextract/), [wget](https://pypi.org/project/wget/), [NLTK](https://pypi.org/project/nltk/), [scikit-learn](https://pypi.org/project/scikit-learn/), [transformers](https://pypi.org/project/transformers/), [SPARQLWrapper](https://pypi.org/project/SPARQLWrapper/), and [python-levenshtein](https://pypi.org/project/python-Levenshtein/).

## Word Embeddings

Word embeddings are required to run the Bi-LSTM-CRF named entity recognition model. 
There are two options for getting an word embedding:
1. Use a publicly available one: `wikipedia-pubmed-and-PMC-w2v.bin` from http://evexdb.org/pmresources/vec-space-models/ 
2. Train a new one, for instance, on the PMC OA subset: JATS files are available from https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/
and can be extracted and preprocessed with articlenizer:
```shell
parse_JATS --in-path path/PMC_OA_JATS_folder/ --out-path path/PMC_OA_Text_folder --ncores 60
articlenizer_prepro --in-path path/PMC_OA_Text_folder --out-path path/PMC_OA_prepro --ncores 60 
train_word_emb --in-path path/PMC_OA_prepro --out-path data/word_embs/ --ncores 60
```

## Data transform and split

As data input format BRAT standoff-format is assumed.
It needs to be first transformed into data suited for the models described below.
This can be done by:
```shell
brat_to_bio --in-path data/minimal_example/text/ --out-path data/minimal_example/bio
```
The data also needs to be split into training development and test set. 
```
split_data --in-path data/minimal_example/bio/ --out-path data/minimal_example/
```
and after SoMeSci was downloaded and BRAT folders were extracted/copied to `data`:
```shell
brat_to_bio --in-path data/PLoS_sentences --out-path data/PLoS_sentences_bio --ncores 4

brat_to_bio --in-path data/PLoS_methods --out-path data/PLoS_methods_bio --ncores 4
split_data --in-path data/PLoS_methods_bio --out-path data/PLoS_methods 

brat_to_bio --in-path data/Pubmed_fulltext --out-path data/Pubmed_fulltext_bio --ncores 4
split_data --in-path data/Pubmed_fulltext_bio --out-path data/Pubmed_fulltext 

brat_to_bio --in-path data/Creation_sentences --out-path data/Creation_sentences_bio --ncores 4
split_data --in-path data/Creation_sentences_bio --out-path data/Creation_sentences 
```
Note that PLoS_sentences is entirely used for training and not split

## Named Entity Recognition (NER)

The configuration for training and testing is made in configurations files. Default examples are placed in `configurations/named_entity_recognition`.
After data was preprocessed it can directly be run for the minimal example
```shell
train_model --model-config configurations/named_entity_recognition/minimal_example_LSTM.json --data-config configurations/named_entity_recognition/minimal_example_data.json --feature-file-ext ''
```
and SoMeSci
```shell
train_model --model-config configurations/named_entity_recognition/SoMeSci_LSTM.json --data-config configurations/named_entity_recognition/SomeSci_data_software.json --feature-file-ext ''
```
For SoMeSci different configurations are available to recognize different entities. Check `configurations/named_entity_recognition` and `configurations/data_transforms`.

## Relation Extraction (RE)

Configurations for RE are made in `configurations/relation_extraction` and can also be run directly: 
```shell
train_relext --model-config configurations/relation_extraction/SoMeSci_base_config.json --data-config configurations/relation_extraction/minimal_example_data_config.json
```

## Entity Disambiguation (ED)

No configuration is required for ED, as all implementations are static for now.
To run it a little more data pre-processing is required.
For the minimal example:
```shell
somesci_disambiguation_input --in-paths data/minimal_example/text/ --out-path data
```
and for SoMeSci:
```shell
somesci_disambiguation_input --in-paths data/PLoS_methods data/Pubmed_fulltext data/PLoS_sentences data/Creation_sentences --out-path data

```
as well as some supervised data gathered from [DBpedia](https://www.dbpedia.org/):
```shell
load_dbpedia_info --out-path data/dbpedia
```
Now the disambiguation can be run for the minimal example:
```shell
entity_disambiguation --in-file data/entity_linking_input.json --gold-standard data/minimal_example/linking_gold_standard.json --dbpedia data/dbpedia/dbpedia_software_long.csv.gz
```
and SoMeSci:
```shell
entity_disambiguation --in-file data/entity_linking_input.json --gold-standard data/Linking/artifacts.json --dbpedia data/dbpedia/dbpedia_software_long.csv.gz
```
`Linking/artifacts.json` is contained in SoMeSci.
