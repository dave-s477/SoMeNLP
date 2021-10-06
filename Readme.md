# SoMeNLP

SoMeNLP provides functionality for performing information extraction for software mentions in scientific articles. 
Implemented are: Named Entity Recognition, Relation Extraction and Entity Disambiguation. 

Up to now it has been applied on the SoMeSci dataset (available from [zenodo](https://zenodo.org/record/4701764) or [github](https://github.com/dave-s477/SoMeSci)). 

**The SoMeSci pipeline is adjusted to perform large scale information extraction on PMC for the construction of SoftwareKG in this branch. 
While all code is already included, the documentation is not yet up to date and will be further extended.**

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
