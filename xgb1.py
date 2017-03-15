#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:22:30 2017

@author: antoinemovschin
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import log_loss
import multiprocessing
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse


# nb of processor cores
n_cores = multiprocessing.cpu_count()
# data path
data_path = '../input'
# output files path
output_files_path = '../output'
# random seed
seed = 123
# validation set size
valid_size = .3
# CV parameters
cv_n_splits = 3
cv_test_size = .3

# load the data
LOAD_DATA = True
# create submission file
CREATE_SUBMISSION_FILE = True





##################################################################
# Loading the data
##################################################################
if LOAD_DATA == True:
    df = pd.read_json(open("../input/train.json", "r"))
    
num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",]
cat_feats = ["display_address", "manager_id", "building_id", "street_address"]
misc_feats = ["created", "display_address", "features", "photos"]


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
    
    for f in cat_feats:
        if df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[f].values))
            df[f] = lbl.transform(list(df[f].values))
            
    return df


df = add_features(df)

features_to_use = num_feats
features_to_use.extend(["num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"])
features_to_use.extend(cat_feats)

def to_csr_mat(df):
    tfidf = CountVectorizer(stop_words='english', max_features=200)
    feat_sparse = tfidf.fit_transform(df["features"])
    X = sparse.hstack([df[features_to_use], feat_sparse]).tocsr()
    #desc_sparse = tfidf.fit_transform(df["description"])
    #X = sparse.hstack([df[features_to_use], feat_sparse, desc_sparse]).tocsr()
    return X

X = to_csr_mat(df)
#X = df[features_to_use]
target_num_map = {'high':0, 'medium':1, 'low':2}
y = np.array(df['interest_level'].apply(lambda x: target_num_map[x]))

X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=valid_size, random_state=seed)




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
    param['subsample'] = 0.6
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
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
cv_scores = []
kf = KFold(n_splits=3, shuffle=True, random_state=seed)
for dev_index, val_index in kf.split(range(X_train.shape[0])):
    dev_X, val_X = X_train[dev_index,:], X_train[val_index,:]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    preds, model = runXGB(dev_X, dev_y, val_X, val_y)
    cv_scores.append(log_loss(val_y, preds))
    print(cv_scores)


#cv = ShuffleSplit(n_splits=cv_n_splits, test_size=cv_test_size, random_state=seed)
#cv_score = cross_val_score(clf, X_train, y_train, cv=cv)

preds, model = runXGB(X_train, y_train, X_val, num_rounds=400)
val_score = log_loss(y_val, preds)

print('CV score: log_loss = ' + str(np.mean(cv_scores)))
print('Validation set: log_loss = ' + str(val_score))





##################################################################
# creating submission file
##################################################################
if CREATE_SUBMISSION_FILE == True:
    if LOAD_DATA == True:
        test_df = pd.read_json(open("../input/test.json", "r"))
    test_df = add_features(test_df)
    X_test = to_csr_mat(test_df)
    #X_test = test_df[features_to_use]
    preds, model = runXGB(X_train, y_train, X_test, num_rounds=400)
    out_df = pd.DataFrame(preds)
    out_df.columns = ["high", "medium", "low"]
    out_df["listing_id"] = test_df.listing_id.values
    out_df.to_csv("../output/xgb_starter_2.csv", index=False)
    