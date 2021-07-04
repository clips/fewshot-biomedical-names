from reach import Reach
from tqdm import tqdm
import numpy as np
import fasttext

#############################################################
#############################################################
##################      Sampling      #######################
#############################################################
#############################################################

# fasttext_model_path = '/home/fivez/disorder_linking/data/embeddings/pub2fast.bin'


class Vectorize:

    def __init__(self, fasttext_model_path):

        self.fasttext_model_path = fasttext_model_path
        print('Loading fastText model...')
        self.fasttext_model = fasttext.FastText.load_model(self.fasttext_model_path)
        print('Done')

        self.pretrained_name_embeddings = None
        self.construct_oov = False

    def allow_construct_oov(self):

        if not self.construct_oov:
            self.construct_oov = True

    def vectorize_string(self, string, norm):

        tokens = string.split()
        token_embeddings = []
        for token in tokens:
            vector = self.fasttext_model.get_word_vector(token)
            if norm:
                vector = Reach.normalize(vector)
            token_embeddings.append(vector)
        token_embeddings = np.array(token_embeddings)

        return token_embeddings

    def create_reach_object(self, names, outfile=''):

        names = sorted(names)

        vectors = []
        for name in tqdm(names):
            token_embs = self.vectorize_string(name, norm=False)
            vector = np.average(np.array(token_embs), axis=0)
            vectors.append(vector)

        reach_object = Reach(vectors, names)

        if outfile:
            reach_object.save_fast_format(outfile)

        return reach_object
