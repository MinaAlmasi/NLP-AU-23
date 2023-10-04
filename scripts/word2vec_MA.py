# from https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
# hp data from https://github.com/amephraim/nlp/blob/master/texts/J.%20K.%20Rowling%20-%20Harry%20Potter%204%20-%20The%20Goblet%20of%20Fire.txt

import pathlib

from gensim import utils
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, corpus_paths, tokenizer=None):
        if isinstance(corpus_paths, str):
            # If a single string is provided, convert it to a list for uniform handling
            self.corpus_paths = [corpus_paths]
        elif isinstance(corpus_paths, list):
            self.corpus_paths = corpus_paths
        else:
            raise ValueError("corpus_paths must be a string or a list of strings")
    
    def __iter__(self):
        for corpus_path in self.corpus_paths:
            for line in open(corpus_path, "r", encoding="latin1"):
                yield utils.simple_preprocess(line)

def main(): 
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "harrypotter"

    paths = [file for file in datapath.iterdir() if file.is_file()]

    sentences = MyCorpus(corpus_paths=paths)

    model = Word2Vec(
                    sentences=sentences, 
                    vector_size=300, 
                    sample = 0.001, # downsample frequent words, 
                    window=5, # context window
                    seed=129, # reproducibility
                    workers=1 # reproducibility
                    )

    print(model.wv.most_similar("hermione"))

if __name__ == "__main__":
    main()