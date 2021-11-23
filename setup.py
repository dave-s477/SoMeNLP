import sys
if sys.version_info <= (3,7):
    sys.exit('Python >= 3.7 is required')

from setuptools import setup

def readme():
    with open('Readme.md') as f:
        return f.read()

setup(name='SoMeNLP',
      version='0.1',
      description='NLP procedures for scientific articles and software extraction.',
      long_description=readme(),
      classifiers=[
        'License :: OSI Approved :: GPLv3',
        'Programming Language :: Python :: 3.7',
        'Topic :: NER :: NLP',
      ],
      keywords='scientific entity relation software paper article',
      url='https://github.com/dave-s477/SoMeNLP',
      author='David Schindler',
      author_email='david.schindler@uni-rostock.de',
      license='GPLv3',
      packages=['somenlp'],
      scripts=[
        'bin/train_word_emb',
        'bin/distant_supervision',
        'bin/custom_feature_gen',
        'bin/train_model',
        'bin/tune_model',
        'bin/tune_relext',
        'bin/predict',
        'bin/split_data',
        'bin/train_relext',
        'bin/load_dbpedia_info',
        'bin/entity_disambiguation',
        'bin/somesci_disambiguation_input',
        'bin/map_unique_names_to_files',
        'bin/predict_relext',
        'bin/combine_annotations',
        'bin/generate_file_list'
      ],
      install_requires=[
        'pytest',
        'gensim>=4.0.1',
        'torch',
        'tensorboard',
        'pandas',
        'numpy',
        'beautifulsoup4',
        'wiktextract',
        'wget', 
        'NLTK',
        'scikit-learn',
        'transformers==4.6.1',
        'SPARQLWrapper',
        'python-levenshtein',
        'articlenizer @ https://github.com/dave-s477/articlenizer/tarball/master#egg=package'
      ],
      include_package_data=True,
      zip_safe=False)
