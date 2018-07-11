import datetime

import logging

from feature import feature_extractor as fe
import nlp
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class FeatureVectorizerTDBaseline():
    def __init__(self):
        super().__init__()
        self.ngram_vectorizer = TfidfVectorizer(
            # feature = sklearn.feature_extraction.text.CountVectorizer(
            tokenizer=nlp.tokenize,
            preprocessor=nlp.preprocess,
            ngram_range=(1, 3),
            stop_words=nlp.stopwords,  # We do better when we keep stopwords
            use_idf=True,
            smooth_idf=False,
            norm=None,  # Applies l2 norm smoothing
            decode_error='replace',
            max_features=10000,
            min_df=5,
            max_df=0.501
        )
        self.pos_vectorizer = TfidfVectorizer(
            # feature = sklearn.feature_extraction.text.CountVectorizer(
            tokenizer=None,
            lowercase=False,
            preprocessor=None,
            ngram_range=(1, 3),
            stop_words=None,  # We do better when we keep stopwords
            use_idf=False,
            smooth_idf=False,
            norm=None,  # Applies l2 norm smoothing
            decode_error='replace',
            max_features=5000,
            min_df=5,
            max_df=0.501,
        )

    def transform_inputs(self, tweets_original, tweets_cleaned, out_folder, flag):
        """"
        This function takes a list of tweets, along with used to
        transform the tweets into the format accepted by the model.

        Each tweet is decomposed into
        (a) An array of TF-IDF scores for a set of n-grams in the tweet.
        (b) An array of POS tag sequences in the tweet.
        (c) An array of features including sentiment, vocab, and readability.

        Returns a pandas dataframe where each row is the set of features
        for a tweet. The features are a subset selected using a Logistic
        Regression with L1-regularization on the training data.

        """
        # Features group 1: tfidf weighted n-grams
        td_tfidf = fe.get_ngram_tfidf(self.ngram_vectorizer, tweets_cleaned, out_folder, flag)

        # Features group 2: PoS for ngrams
        # Features group 2: PoS for ngrams
        td_pos=fe.get_ngram_pos_tfidf(self.pos_vectorizer, tweets_cleaned, out_folder, flag)

        # Features group 3: other features
        logger.info("\tgenerating other feature vectors, {}".format(datetime.datetime.now()))
        td_otherfeats = fe.get_oth_features(tweets_original, tweets_cleaned,out_folder)

        '''CHASE basic features={}'''
        logger.info("\tgenerating CHASE hashtag feature vectors, {}".format(datetime.datetime.now()))
        c_hashtags=fe.get_hashtags_in_tweets(tweets_original, out_folder)
        logger.info("\tgenerating CHASE other stats feature vectors, {}".format(datetime.datetime.now()))
        c_stats=fe.get_stats_features(tweets_original, tweets_cleaned, out_folder)


        logger.info("\t\tcompleted, {}, {}".format(td_otherfeats[0].shape,datetime.datetime.now()))

        # Now concatenate all features in to single sparse matrix
        M = np.concatenate([td_tfidf[0], td_pos[0], td_otherfeats[0],
                            c_hashtags[0],c_stats[0]], axis=1)
        #print(M.shape)
        features_by_type={}
        features_by_type[fe.NGRAM_FEATURES_VOCAB]=td_tfidf
        features_by_type[fe.NGRAM_POS_FEATURES_VOCAB]=td_pos
        features_by_type[fe.TWEET_TD_OTHER_FEATURES_VOCAB]=td_otherfeats
        features_by_type[fe.TWEET_HASHTAG_FEATURES_VOCAB]=c_hashtags
        features_by_type[fe.TWEET_STATS_FEATURES_VOCAB]=c_stats
        return [pd.DataFrame(M), features_by_type]
