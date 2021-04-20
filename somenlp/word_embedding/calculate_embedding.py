
import os
import gensim
import random
import logging

from pathlib import Path

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class SentenizedInput(object):
    """Simple linebased iterator for data. 
    Assumes sentences are split by newline character
    and tokens are split by whitespaces. 
    """
    def __init__(self, file_list, seed):
        random.seed(seed)
        random.shuffle(file_list)
        self.file_list = file_list
 
    def __iter__(self):
        for f in self.file_list:
            with f.open() as text:
                for line in text:
                    if line.strip():
                        yield line.split()

def resume_training_embedding(checkpoint, in_files, out_path, epochs, replace=False, format=2, seed=42, ncores=8):
    """Re-train a Word2Vec embedding from a gensim model checkpoint

    Args:
        checkpoint (string): path to pre-trained model
        in_files (list of PosixPaths): files of training corpus 
        out_path (string): output path
        epochs (int): number of training epochs
        replace (bool, optional): removes the original checkpoint. Defaults to False.
        format (int, optional): determines the output format between bin/gensim-model/both. Defaults to 2.
        seed (int, optional): data shuffling seed. Defaults to 42.
        ncores (int, optional): number of gensim workers. Defaults to 8.
    """
    iterator = SentenizedInput(in_files, seed)
    model = gensim.models.Word2Vec.load(checkpoint)
    model.build_vocab(iterator, update=True)
    model.train(iterator, total_examples=model.corpus_count, epochs=epochs)
    out_loc = checkpoint.split('.model')[0] + "_re-{}".format(epochs)
    if format == 0:
        print("Saving model as bin")
        model.wv.save_word2vec_format(out_loc + '.bin', binary=True)
    elif format == 1:
        print("Saving trainable model")
        model.save(out_loc + '.model')
    else:
        print("Saving model as bin and trainable")
        model.wv.save_word2vec_format(out_loc + '.bin', binary=True)
        model.save(out_loc + '.model')

    if replace:
        os.remove(checkpoint)

def train_embedding(name, in_files, out_path, emb_dim=200, win_size=5, min_count=5, epochs=1, format=2, seed=42, ncores=8):
    """Train a gensim Word2Vec embedding on a corpus

    Args:
        name (string): name for writing the model 
        in_files (list of PosixPaths): files of training corpus 
        out_path (string): output path
        emb_dim (int, optional): size of the embedding. Defaults to 200.
        win_size (int, optional): w2v window size. Defaults to 200.
        min_count (int, optional): word mincount to be in embedding. Defaults to 200.
        epochs (int, optional): number of training epochs
        format (int, optional): determines the output format between bin/gensim-model/both. Defaults to 2.
        seed (int, optional): data shuffling seed. Defaults to 42.
        ncores (int, optional): number of gensim workers. Defaults to 8.
    """
    iterator = SentenizedInput(in_files, seed)
    model = gensim.models.Word2Vec(iterator, size=emb_dim, window=win_size, min_count=min_count, workers=ncores, sg=1, iter=epochs)

    out_loc = out_path + '/' + name 
    if format == 0:
        print("Saving model as bin")
        model.wv.save_word2vec_format(out_loc + '.bin', binary=True)
    elif format == 1:
        print("Saving trainable model")
        model.save(out_loc + '.model')
    else:
        print("Saving model as bin and trainable")
        model.wv.save_word2vec_format(out_loc + '.bin', binary=True)
        model.save(out_loc + '.model')
