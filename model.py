# -*- coding: UTF-8 -*-
from __future__ import division
__author__ = 'shenxiangxiang@gmail.com'
import os
from keras.engine.training import Model
from keras.layers import Input,LSTM, Dense, Dropout
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import numpy as np
from helper import get_news_representation

nb_hidden_units = 200
dropout = 0.2
l2_norm_alpha = 0.0001

class FinancialNewsAnalysisModel(object):
    model = None
    def __init__(self, nb_time_step, dim_data, batch_size=1, model_path=None):
        self.model_path = model_path
        self.model_path = model_path
        self.batch_size = batch_size
        self.size_of_input_data_dim = dim_data
        self.size_of_input_timesteps = nb_time_step
        self.build()
        self.weight_loaded = False
        if model_path is not None:
            self.load_weights()

    def build(self):
        dim_data = self.size_of_input_data_dim
        nb_time_step = self.size_of_input_timesteps
        news_input = Input(shape=(nb_time_step, dim_data))
        lstm = LSTM(output_dim=nb_hidden_units, dropout_U=dropout, dropout_W=dropout,
                            W_regularizer=l2(l2_norm_alpha), b_regularizer=l2(l2_norm_alpha), activation='tanh')
        bi_lstm = Bidirectional(lstm, input_shape=(nb_time_step, dim_data), merge_mode='concat')
        all_news_rep = bi_lstm(news_input)
        news_predictions = Dense(1)(all_news_rep)
        self.model = Model(news_input, news_predictions, name="deep rnn for financial news analysis")

    def reset(self):
        for l in self.model.layers:
            if type(l) is LSTM:
                l.reset_status()

    def compile_model(self, lr=0.0001, arg_weight=1.):
        optimizer = Adam(lr=lr)
        loss = 'mse'
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit_model(self, X, y, X_val = None, y_val = None, epoch=300):
        early_stopping = EarlyStopping(monitor='val_loss',patience=3, verbose=0)
        if X_val is None:
            self.model.fit(X, y, batch_size=self.batch_size, nb_epoch=epoch, validation_split=0.2,
                           shuffle=True, callbacks=[early_stopping])
        else:
            self.model.fit(X, y, batch_size=self.batch_size, nb_epoch=epoch, validation_data=(X_val, y_val),
                           shuffle=True, callbacks=[early_stopping])


    def save(self):
        self.model.save_weights(self.model_path, overwrite=True)

    def load_weights(self):
        if os.path.exists(self.model_path):
            self.model.load_weights(self.model_path)
            self.weight_loaded = True

    def print_weights(self, weights=None, detail=False):
        weights = weights or self.model.get_weights()
        for w in weights:
            print("w%s: sum(w)=%s, ave(w)=%s" % (w.shape, np.sum(w), np.average(w)))
        if detail:
            for w in weights:
                print("%s: %s" % (w.shape, w))

    def model_eval(self, X, y):
        y_hat = self.model.predict(X, batch_size=1)
        count_true = 0
        count_all = y.shape[0]
        for i in range(y.shape[0]):
            count_true = count_true + 1 if y[i,0]*y_hat[i,0]>0 else count_true
            print y[i,0],y_hat[i,0]
        print count_all,count_true

if __name__ == '__main__':
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    training_set_path = os.path.join(this_file_path,"newsdata",)
    file_list = os.listdir(training_set_path,)
    X = None
    y = None
    for idx, f in enumerate(file_list):
        file_path = os.path.join(training_set_path, f)
        (X_tmp, y_tmp) = get_news_representation(file_path)
        X = X_tmp if X is None else np.vstack((X, X_tmp))
        y = y_tmp if y is None else np.vstack((y, y_tmp))
    y = np.resize(y,(y.shape[0], 1))


    training_set_path = os.path.join(this_file_path,"testdata",)
    file_list = os.listdir(training_set_path,)
    X_val = None
    y_val = None
    for idx, f in enumerate(file_list):
        file_path = os.path.join(training_set_path, f)
        (X_tmp, y_tmp) = get_news_representation(file_path,year='2016年')
        X_val = X_tmp if X_val is None else np.vstack((X_val, X_tmp))
        y_val = y_tmp if y_val is None else np.vstack((y_val, y_tmp))
    y_val = np.resize(y_val,(y_val.shape[0], 1))

    fa_model = FinancialNewsAnalysisModel(200,100,batch_size=32,model_path="fa.model.weights")
    fa_model.compile_model()

    fa_model.fit_model(X_val, y_val,X,y)
    fa_model.save()
    fa_model.model_eval(X_val, y_val)
