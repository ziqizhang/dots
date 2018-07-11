#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import datetime
import logging
import sys
import os

import numpy
import pandas as pd
from sklearn import svm
from sklearn.cross_validation import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier, LogisticRegression

from feature.feature_td_baseline import FeatureVectorizerTDBaseline
from ml import ml_util
import nlp

#0=min/max; 1=std
SCALING_STRATEGY = 0
FEATURE_SELECTION = True

#####################################################


class Classifier(object):
    """
    """

    def __init__(self,
                 folder_sysout,
                 output_scores_per_dataset=False):
        self.raw_data = numpy.empty
        self.cleaned_data = numpy.empty
        self.sys_out = folder_sysout  # exclusive 16
        self.output_scores_per_ds = output_scores_per_dataset

    def load_data(self, data_file):
        print("loading input data from: {}, exist={}".format(data_file,
                                                                   os.path.exists(data_file)))
        self.raw_data = pd.read_csv(data_file, sep=',', encoding="utf-8")
        self.cleaned_data = pd.read_csv(data_file + "c.csv", sep=',', encoding="utf-8")

    # algorithm: 0=sgd, 1=LR, 2=random forest, 3=lsvm, 4=rbf svm
    def cross_val(self, identifier, algorithm, nfold):
        print(datetime.datetime.now())
        meta_M = self.feature_extraction(self.raw_data.tweet, self.sys_out,
                                         self.cleaned_data.tweet)
        X_train_data = meta_M[0]
        y_train = self.raw_data['class']
        X_indexes = list(self.raw_data.index.values)
        # M=self.feature_scale(M)

        X_train_data = ml_util.feature_scale(SCALING_STRATEGY, X_train_data)
        y_train = y_train.astype(str)

        instance_data_source_column = None
        accepted_ds_tags = None
        if self.output_scores_per_ds:
            instance_data_source_column = pd.Series(self.raw_data.ds)
            accepted_ds_tags = None

        ######################### SGDClassifier #######################
        if algorithm == 0:
            self.learn_general(nfold, identifier, "sgd",
                               X_train_data, y_train,X_indexes,
                               self.sys_out,
                               FEATURE_SELECTION,
                               instance_data_source_column, accepted_ds_tags)

        ######################### Stochastic Logistic Regression#######################
        elif algorithm == 1:
            self.learn_general(nfold, identifier, "lr",
                               X_train_data, y_train,X_indexes,
                               self.sys_out,
                               FEATURE_SELECTION,
                               instance_data_source_column, accepted_ds_tags)

        ######################### Random Forest Classifier #######################
        elif algorithm == 2:
            self.learn_general(nfold, identifier, "rf",
                               X_train_data, y_train,X_indexes,
                               self.sys_out,
                               FEATURE_SELECTION,
                               instance_data_source_column, accepted_ds_tags)

        ###################  liblinear SVM ##############################
        elif algorithm == 3:
            self.learn_general(nfold, identifier, "svml",
                               X_train_data, y_train,X_indexes,
                               self.sys_out,
                               FEATURE_SELECTION,
                               instance_data_source_column, accepted_ds_tags)

        ##################### RBF svm #####################
        elif algorithm == 4:
            self.learn_general(nfold, identifier, "svmrbf",
                               X_train_data, y_train,X_indexes,
                               self.sys_out,
                               FEATURE_SELECTION,
                               instance_data_source_column, accepted_ds_tags)
            print("complete, {}".format(datetime.datetime.now()))

    def create_classifier(self, outfolder, model_label, task_identifier):
        classifier = None
        model_file = None
        cl_tuning_params = None
        subfolder = outfolder + "/models"
        try:
            os.stat(subfolder)
        except:
            os.mkdir(subfolder)

        if (model_label == "rf"):
            print("== Random Forest ...{}".format(datetime.datetime.now()))
            classifier = RandomForestClassifier(n_estimators=20, n_jobs=-1)
            cl_tuning_params = {}
            model_file = subfolder + "/%s.m" % task_identifier
        if (model_label == "svml"):
            cl_tuning_params = {}
            print("== SVM, kernel=linear ...{}".format(datetime.datetime.now()))
            classifier = svm.LinearSVC(class_weight='balanced', C=0.01, penalty='l2', loss='squared_hinge',
                                       multi_class='ovr')
            model_file = subfolder + "/%s.m" % task_identifier

        if (model_label == "svmrbf"):
            cl_tuning_params = {}
            print("== SVM, kernel=rbf ...{}".format(datetime.datetime.now()))
            classifier = svm.SVC()
            model_file = subfolder + "/%s.m" % task_identifier

        if (model_label == "sgd"):
            print("== SGD ...{}".format(datetime.datetime.now()))
            # DISABLED because grid search takes too long to complete
            cl_tuning_params = {}
            classifier = SGDClassifier(loss='log', penalty='l2', n_jobs=-1)
            model_file = subfolder + "/%s.m" % task_identifier
        if (model_label == "lr"):
            print("== Stochastic Logistic Regression ...{}".format(datetime.datetime.now()))
            cl_tuning_params = {}
            classifier = LogisticRegression(random_state=111)
            model_file = subfolder + "/%s.m" % task_identifier

        return classifier, model_file

    def learn_general(self, nfold, task, model_label, X_train, y_train, X_indexes,
                      outfolder, feature_selection=False,
                      instance_data_source_tags=None, accepted_ds_tags: list = None
                      ):
        if feature_selection:
            select = SelectFromModel(LogisticRegression(class_weight='balanced', penalty="l1", C=0.01))
            X_train = select.fit_transform(X_train, y_train)

        cls, model_file = self.create_classifier(outfolder, model_label, task)
        nfold_predictions = cross_val_predict(cls, X_train, y_train, cv=nfold)

        ml_util.save_scores(nfold_predictions, y_train, None, None,
                            X_indexes, #nfold index
                            None, #heldout index
                            model_label, task,
                            2, outfolder,
                            instance_data_source_tags, accepted_ds_tags)

    def feature_extraction(self, raw_data_column, sysout, cleaned_data_column=None):
        tweets = raw_data_column
        tweets = [x for x in tweets if type(x) == str]
        print("FEATURE EXTRACTION AND VECTORIZATION FOR ALL data, insatance={}, {}"
                    .format(len(tweets), datetime.datetime.now()))
        print("\tbegin feature extraction and vectorization...")

        if cleaned_data_column is None:
            tweets_cleaned = [nlp.strip_hashtags(x) for x in tweets]
        else:
            tweets_cleaned = [x for x in cleaned_data_column if type(x) == str]
        # tweets_cleaned = [text_preprocess.preprocess_clean(x, True, True) for x in tweets]
        fv = FeatureVectorizerTDBaseline()
        M = fv.transform_inputs(tweets, tweets_cleaned, sysout, "na")
        print("FEATURE MATRIX dimensions={}".format(M[0].shape))
        return M


if __name__ == '__main__':
    #arg1=identifier; arg2=input data csv; arg3=outfolder, arg4 (optional) True or False to indicate
    #if scores should show for each sub-dataset (in case the dataset is mixed)
    nfold=10
    classifier = Classifier(sys.argv[2], bool(sys.argv[3]))

    classifier.load_data(sys.argv[1])
    classifier.cross_val("svml-rm-10fold", 3, 10)
