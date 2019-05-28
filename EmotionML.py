#=====================#
# Author: Shiyun Yang #
#=====================#

"""
class EmotionML: 
    - load EEG data
    - clean data: remove missing values
    - divide data into time series sequences
    - extract feature from time series using tsfresh package
    - classify emotion using machine learning ensemble
"""

# import libraries
import os, glob, copy, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import json
import pickle
from statistics import mean 
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn import model_selection
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

from voting import VotingClassifier

MODEL_PATH = './Models/'
MODEL_NAMES = ['./Models/KNeighborsClassifier_model.pkl',
               './Models/DecisionTreeClassifier_model.pkl',
               './Models/RandomForestClassifier_model.pkl',
               './Models/AdaBoostClassifier_model.pkl',
               './Models/GradientBoostingClassifier_model.pkl',
               './Models/GaussianNB_model.pkl',
               './Models/LinearDiscriminantAnalysis_model.pkl',
               './Models/XGBClassifier_model.pkl']

TSFRESH_SETTINGS = MinimalFCParameters()
RAW_COLUMNS = ['attention', 'meditation', 'delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'highGamma']
COLUMNS = ['delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'highGamma']
NEW_COLUMNS = ['id', 'time', 'delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'highGamma']
SEQ_SIZE = 8

class EmotionML(object): 
    def __init__(self): 
        self.models = None
        self.raw = None         # raw data from json.loads
        self.data = None        # pd dataframe
        self.cleaned = None     # data after cleaning
        self.sequences = None
        self.MLInput = None

    def load_data(self, json_data): 
        """Loads data from json and convert to pandas dataframe. """
        loaded = json.loads(json_data)
        self.raw = loaded
        df = pd.DataFrame.from_dict(loaded, orient='index', columns=RAW_COLUMNS)
        self.data = df
        print('Data loaded. ')

    def __clean_df(self, df): 
        """Helper funtion: cleans a dataframe
        Parameter
        ----------
        df: 10 columns (first 2 are attention and meditation, last 8 are brainwave bands)

        Return
        ----------
        df: 8 columns (removed attention and meditation), no NaNs, no 0s
        """
        # check if columns are correct
        assert df.shape[1] == 10, 'ERROR: number of columns is NOT 10'
        assert df.columns[0] == 'attention' and df.columns[1] == 'meditation', \
               "ERROR: headers of first 2 columns are not 'attention' and 'meditation'"
        # drop attention and meditation columns
        df = df.iloc[:, 2:]
        # drop rows with any 0
        df = df[(df != 0).all(1)]
        # drop rows with missing values
        df = df.dropna()
        # reset index
        df.reset_index(drop=True, inplace=True)
        return df

    def _clean_data(self): 
        """Cleans data for preprocessing. """
        df = self.data
        # check if data is loaded
        assert df.empty == False, 'ERROR: data not loaded, please load data first. '
        # clean data
        df = self.__clean_df(df)
        self.cleaned = df
        print('Data cleaning completed. ')

    def __gen_seq(self, df, seq_size): 
        """Helper function: generates sequences from a dataframe. """
        ret = []
        num_seq = int(df.shape[0] / seq_size)
        skip = df.shape[0] % seq_size
        for i in range(num_seq): 
            ret.append([df.iloc[skip+i*seq_size:skip+(i+1)*seq_size, :].values])
            #ret[i].reset_index(drop=True, inplace=True)
        return ret

    def _data2seq(self): 
        """Converts data to sequences for feature extraction. E.g. 40 seconds of data --> 5 sequences * 8 second/sequence"""
        df = self.cleaned
        # check if number of cols is 8
        assert df.shape[1] == 8, 'ERROR: number of columns is NOT 8'
        # generate sequences
        seqs = self.__gen_seq(df, SEQ_SIZE)
        # convert to np array
        self.sequences = np.asarray(seqs)
        print('Data converted to sequences of size {}.'.format(SEQ_SIZE))

    def __format_tsfresh(self, seqs, ret):
        """Helper function: converts sequences to input format of tsfresh"""
        for i in range(seqs.shape[0]): 
            df = pd.DataFrame(data=seqs[i, 0], columns=COLUMNS)
            df = df.assign(id=[i]*SEQ_SIZE)
            df = df.assign(time=df.index)
            df = df.reindex(columns = NEW_COLUMNS)
            ret = ret.append(df)
        return ret

    def preprocess(self): 
        """Extracts features from sequences"""
        self._clean_data()
        self._data2seq()
        # check if sequences is available
        assert self.sequences is not None, 'ERROR: no sequences available, please run data2seq first'

        # format sequences for tsfresh
        formated_seqs = pd.DataFrame(columns=NEW_COLUMNS)
        formated_seqs = self.__format_tsfresh(self.sequences, formated_seqs)
        # reset index and convert to float
        formated_seqs.reset_index(drop=True, inplace=True)
        formated_seqs = formated_seqs.astype(float)

        # extract features
        features = extract_features(formated_seqs, column_id="id", column_sort="time", default_fc_parameters=TSFRESH_SETTINGS)
        self.MLInput = np.array(features)
        print('Data preprocessing completed. ')

    def prob2class(self, prob): 
        if prob <= 0.2: 
            c = 1
        elif prob <= 0.4: 
            c = 2
        elif prob <= 0.6: 
            c = 3
        elif prob <= 0.8: 
            c = 4
        else: 
            c = 5
        return c

    def predict(self): 
        # load models
        knn = pickle.load(open(MODEL_NAMES[0], 'rb'))
        rf = pickle.load(open(MODEL_NAMES[2], 'rb'))
        adaB = pickle.load(open(MODEL_NAMES[3], 'rb'))
        gb = pickle.load(open(MODEL_NAMES[4], 'rb'))
        nb = pickle.load(open(MODEL_NAMES[5], 'rb'))
        lda = pickle.load(open(MODEL_NAMES[6], 'rb'))
        xgb = pickle.load(open(MODEL_NAMES[7], 'rb'))

        voting_clf = VotingClassifier(
                estimators = [('xgb', xgb), ('knn', knn), ('rf', rf), ('gb', gb), ('adaB', adaB), ('nb', nb), ('lda', lda)],
                voting = 'hard')
        #model_predictions = voting_clf._predict(self.MLInput)
        #model_probs = voting_clf._collect_probas(self.MLInput)
        _, display_probs = voting_clf.predict(self.MLInput)
        avg_prob = mean(display_probs)
        final_c = self.prob2class(avg_prob)
        Cs = []
        for prob in display_probs: 
            Cs.append(self.prob2class(prob))
        class_names = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
        ret = {}
        ret['vote0'] = [round(avg_prob*100), final_c, class_names[final_c-1]]
        ret['vote1'] = [round(display_probs[0]*100), Cs[0], class_names[Cs[0]-1]]
        ret['vote2'] = [round(display_probs[1]*100), Cs[1], class_names[Cs[1]-1]]
        ret['vote3'] = [round(display_probs[2]*100), Cs[2], class_names[Cs[2]-1]]
        ret['vote4'] = [round(display_probs[3]*100), Cs[3], class_names[Cs[3]-1]]
        ret['vote5'] = [round(display_probs[4]*100), Cs[4], class_names[Cs[4]-1]]
        ret['feat_time'] = list(self.raw.keys())
        ret['attention'] = self.data['attention'].tolist()
        ret['meditation'] = self.data['meditation'].tolist()
        ret['delta'] = self.data['delta'].tolist()
        ret['theta'] = self.data['theta'].tolist()
        ret['lowAlpha'] = self.data['lowAlpha'].tolist()
        ret['highAlpha'] = self.data['highAlpha'].tolist()
        ret['lowBeta'] = self.data['lowBeta'].tolist()
        ret['highBeta'] = self.data['highBeta'].tolist()
        ret['lowGamma'] = self.data['lowGamma'].tolist()
        ret['highGamma'] = self.data['highGamma'].tolist()

        return ret

