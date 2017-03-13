# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:31:45 2017

@author: antoinemovschin
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, KFold
from sklearn.metrics import log_loss
import multiprocessing

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



##################################################################
# Adding features
##################################################################
def add_features(df):
    df["num_photos"] = df["photos"].apply(len)
    df["num_features"] = df["features"].apply(len)
    df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
    df["created"] = pd.to_datetime(df["created"])
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    return df



df = add_features(df)
num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"]
X = df[num_feats]
y = df["interest_level"]


X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=valid_size, random_state=seed)




##################################################################
# Cross validation
##################################################################
clf = RandomForestClassifier(
        n_estimators=1000, 
        criterion="gini",       # default
        max_features=None,      # default
        max_depth=None,         # default
        min_samples_split=2,    # default
        min_samples_leaf=1,     # default
        n_jobs=n_cores-1,       # nb of cores (leave one for OS)
        random_state=seed,
        warm_start=False,        # improves last fit at each iteration
        verbose=1)

cv_scores = []
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for dev_index, val_index in kf.split(range(X_train.shape[0])):
    dev_X = X_train.values[dev_index,:] 
    val_X = X_train.values[val_index,:]
    dev_y = y_train.values[dev_index]
    val_y = y_train.values[val_index]
    clf.fit(dev_X, dev_y)
    preds = clf.predict_proba(val_X)
    cv_scores.append(log_loss(val_y, preds))
    print(cv_scores)


#cv = ShuffleSplit(n_splits=cv_n_splits, test_size=cv_test_size, random_state=seed)
#cv_score = cross_val_score(clf, X_train, y_train, cv=cv)

y_val_pred = clf.predict_proba(X_val)
val_score = log_loss(y_val, y_val_pred)

print('CV score: log_loss = ' + str(np.mean(cv_scores)))
print('Validation set: log_loss = ' + str(val_score))


##################################################################
# creating submission file
##################################################################
if CREATE_SUBMISSION_FILE == True:
    df = pd.read_json(open("../input/test.json", "r"))
    print(df.shape)
    df = add_features(df)
    X = df[num_feats]
    y = clf.predict_proba(X)
    
    labels2idx = {label: i for i, label in enumerate(clf.classes_)}
    sub = pd.DataFrame()
    sub["listing_id"] = df["listing_id"]
    for label in ["high", "medium", "low"]:
        sub[label] = y[:, labels2idx[label]]
    sub.to_csv("../output/submission_rf.csv", index=False)
