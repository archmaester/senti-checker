from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
import numpy as np


class Tweet2Vec:
    def __init__(self):
        self._dim = 50
        self._doc2vec_model = Doc2Vec(vector_size=self._dim, epochs=100, min_count=1)

    def train(self, tweets):
        tagged_tweets = list(self.__tag_tweet(tweets))
        self._doc2vec_model.build_vocab(tagged_tweets)
        self._doc2vec_model.train(tagged_tweets,
                                  total_examples=self._doc2vec_model.corpus_count,
                                  epochs=self._doc2vec_model.epochs)

    def __tag_tweet(self, tweets):
        for i in range(len(tweets)):
            yield TaggedDocument(simple_preprocess(tweets[i]), [i])

    def embed_tweets(self, tweets):
        embeddings = np.zeros((len(tweets), self._dim))
        for i in range(len(tweets)):
            embedding = self._doc2vec_model.infer_vector(simple_preprocess(tweets[i]))
            embeddings[i, :] = embedding
        return embeddings