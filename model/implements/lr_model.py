#!/usr/bin/env python
# -*- encoding: utf8 -*-

__author__ = 'Michael-Wang'

from sklearn.linear_model import LogisticRegression


class LrModel:
    def __init__(self):
        self.model = LogisticRegression(solver='sag',
                                        tol=1e-3,
                                        max_iter=1000,
                                        random_state=42,
                                        multi_class='multinomial',
                                        # verbose=10,
                                        n_jobs=1)

    def train(self, X_train, y_train):
        print 'train LR model start....'
        self.model.fit(X_train, y_train)
        # self.print_feature_importance()
        print 'train LR model finished'
        print '======================='

    def predict(self, x):
        return self.model.predict_proba(x)

    def print_feature_importance(self):
        # columns = vec.get_feature_names()
        # feature_coef_list = sorted(zip(columns, self.model.coef_[0]), key=lambda one: one[1], reverse=True)
        # for element in feature_coef_list:
        #     print element
        pass

    def save_model(self):
        # save_pickle(self.model, MODEL_SAVE_PATH + "lr/model.pkl")
        pass