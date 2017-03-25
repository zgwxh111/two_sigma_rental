#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:22:30 2017

@author: antoinemovschin
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, KFold
from sklearn.metrics import log_loss
import multiprocessing
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse
import pickle # to save models into files
import mypy1 as mp # custom functions


# nb of processor cores
n_cores = multiprocessing.cpu_count()
# data path
data_path = '../input/'
# output files path
output_path = '../output/'
models_path = output_path + 'models/'
predictions_path = output_path + 'predictions/'
# random seed
seed = 123
# validation set size
valid_size = .0
# CV parameters
cv_n_splits = 3
cv_test_size = .3

# load the data
LOAD_DATA = True
DO_CV = True
# create submission file
CREATE_SUBMISSION_FILE = False







##################################################################
# Adding features
##################################################################
def add_features(df):
    df["num_photos"] = df["photos"].apply(len)
    df["num_features"] = df["features"].apply(len)
    df['features'] = df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
    df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
    df["created"] = pd.to_datetime(df["created"])
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    df["created_hour"] = df["created"].dt.hour
    
    #df['price'] = df['price'].apply(np.log)
    return df

def encode_labels (df, lbl_dict, cat_feats):
    for f in cat_feats:
        if df[f].dtype=='object':
            #print(f)
            lbl = lbl_dict[f]
            #lbl = preprocessing.LabelEncoder()
            #lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            val = list(df[f].values)
            #val = [x if x in lbl.classes_ else '_unknown_' for x in df[f].values]
            # or alternatively: 
            #val = [(lambda x: x if x in lbl.classes_ else '_unknown_')(x) for x in df[f].values]
            df[f] = lbl.transform(val)
    return df

def to_csr_mat(df1, df2, features_to_use):
    tfidf = CountVectorizer(stop_words='english', max_features=200)
    feat_sparse_1 = tfidf.fit_transform(df1["features"])
    feat_sparse_2 = tfidf.transform(df2["features"])
    X1 = sparse.hstack([df1[features_to_use], feat_sparse_1]).tocsr()
    X2 = sparse.hstack([df2[features_to_use], feat_sparse_2]).tocsr()    
    #desc_sparse = tfidf.fit_transform(df["description"])
    #X = sparse.hstack([df[features_to_use], feat_sparse, desc_sparse]).tocsr()
    return X1, X2


##################################################################
# Training function
##################################################################
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000, verbose=True):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20, verbose_eval=verbose)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds, verbose_eval=verbose)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


##################################################################
# Cross validation 
##################################################################
def crossval (X_train, y_train, Nfolds):
    cv_scores = []
    kf = KFold(n_splits=Nfolds, shuffle=True, random_state=seed)
    for dev_index, val_index in kf.split(range(X_train.shape[0])):
        dev_X, val_X = X_train[dev_index,:], X_train[val_index,:]
        dev_y, val_y = y_train[dev_index], y_train[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
    return cv_scores
    


##################################################################
# Data processing
##################################################################
def create_train_test_sets (preprocess = add_features, features_to_add = None):
    train_df = pd.read_json(open(data_path + "train.json", "r"))
    test_df = pd.read_json(open(data_path + "test.json", "r"))

    num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",]
    cat_feats = ["display_address", "manager_id", "building_id", "street_address"]
    misc_feats = ["created", "display_address", "features", "photos"]
    
    # Creating the label encoders
    lbl_dict = {}
    for f in cat_feats:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
    #        lbl.classes_ = np.append(lbl.classes_, '_unknown_')
            lbl_dict[f] = lbl
    
    train_df = preprocess(train_df)
    train_df = encode_labels(train_df, lbl_dict, cat_feats)
    test_df = preprocess(test_df)
    test_df = encode_labels(test_df, lbl_dict, cat_feats)

    # features to use
    features_to_use = num_feats
    #features_to_use.extend(["price2", "bedrooms2", "bathrooms2"])
    features_to_use.extend(["num_photos", "num_features", "num_description_words",
                 "created_year", "created_month", "created_day",
                 "listing_id", "created_hour"])
    features_to_use.extend(cat_feats)
    if features_to_add is not None:
        features_to_use.extend(features_to_add)
    
    # compressed sparse row matrices
    X_train, X_test = to_csr_mat(train_df, test_df, features_to_use)
    #X = df[features_to_use]
    target_num_map = {'high':0, 'medium':1, 'low':2}
    y_train = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
    
    return X_train, y_train, X_test
    
    

##################################################################
# Output files saving
##################################################################
def create_output_files (X_train, y_train, X_test, cv_scores = None, num_rounds=400):
    # make predictions
    preds, model = runXGB(X_train, y_train, X_test, num_rounds=400)
    out_df = pd.DataFrame(preds)
    out_df.columns = ["high", "medium", "low"]
    out_df["listing_id"] = test_df.listing_id.values
    
    # extension for saved files
    date = mp.strdate()
    extension = date
    if DO_CV == True:
        if cv_scores is not None:
            extension += '_CV_' + str(round(np.mean(cv_scores), 5))

    # save files
    out_df.to_csv(predictions_path + "xgb1" + extension + ".csv", index=False)
    pickle.dump(model, open(models_path + "xgb1" + extension + ".dat", "wb"))





##################################################################
# MAIN
##################################################################


# Load the data
if LOAD_DATA == True:
    X_train, y_train, X_test = create_train_test_sets()
    

# Cross validation
if DO_CV == True:
    cv_scores = crossval(X_train, y_train, 2)
    #cv = ShuffleSplit(n_splits=cv_n_splits, test_size=cv_test_size, random_state=seed)
    #cv_score = cross_val_score(clf, X_train, y_train, cv=cv)
    print('CV score: log_loss = ' + str(np.mean(cv_scores)))
    if (valid_size > 0):
        preds, model = runXGB(X_train, y_train, X_val, num_rounds=400)
        val_score = log_loss(y_val, preds)
        print('Validation set: log_loss = ' + str(val_score))



# creating output file
if CREATE_SUBMISSION_FILE == True:
    create_output_files(X_train, y_train, X_test, cv_scores, num_rounds=400)
    