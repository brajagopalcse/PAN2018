import gensim
from gensim import corpora
from gensim.models import KeyedVectors

import re, sys
import numpy as np


class GensimModel:

    def __init__(self):
        self.LDA = gensim.models.ldamodel.LdaModel
        self.modelGlove200 = KeyedVectors.load('./resources/glove.twitter.27B.200d.model')
        self.modelGlove25 = KeyedVectors.load('./resources/glove.twitter.27B.100d.model')

    def get_vecs(self, list_tweets, list_lemmas, dimension):
        val = np.zeros(dimension)
        for tweet, lemma in zip(list_tweets, list_lemmas):
            if dimension == 100:
                val += self.create_tweet_vectors_100(tweet, lemma)
            elif dimension == 200:
                val += self.create_tweet_vectors_200(tweet, lemma)
            else:
                print('For now only 25 and 200 are available')
                sys.exit(1)
        return val

    def create_tweet_vectors_200(self, word_list, lemma_list):
        hits = 0
        val = np.zeros(200)
        if len(word_list) == 0:
            return val
        for word, lemma in zip(word_list, lemma_list):
            word = word.lower()
            # This is for all type of count
            if word in self.modelGlove200:
                val += self.modelGlove200[word]
                hits += 1
            elif lemma in self.modelGlove200:
                val += self.modelGlove200[lemma]
                hits += 1
        if hits != 0:
            val /= hits
        if hits == 0:
            print('hits is null')
        return val

    def create_tweet_vectors_100(self, word_list, lemma_list):
        hits = 0
        val = np.zeros(100)
        if len(word_list) == 0:
            return val
        for word, lemma in zip(word_list, lemma_list):
            word = word.lower()
            # This is for all type of count
            if word in self.modelGlove25:
                val += self.modelGlove25[word]
                hits += 1
            elif lemma in self.modelGlove25:
                val += self.modelGlove25[lemma]
                hits += 1
        if hits != 0:
            val /= hits
        if hits == 0:
            print('hits is null')
        return val

    def find_topics(self, documents):
        dictionary = corpora.Dictionary(documents)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in documents]
        lda_model = self.LDA(doc_term_matrix, num_topics=3, id2word=dictionary, passes=50)
        topics = []
        for key, topic in lda_model.show_topics():
            topics.append(re.findall('"([^"]*)"', topic))
        return topics
