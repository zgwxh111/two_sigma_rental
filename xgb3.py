#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:22:30 2017

@author: antoinemovschin
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LinearRegression
import multiprocessing
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse
import pickle # to save models into files
from itertools import product

import sys
if '/Users/antoinemovschin/Documents/python/' not in sys.path:
    sys.path.append("/Users/antoinemovschin/Documents/python/")
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
CREATE_SUBMISSION_FILE = True







##################################################################
# Adding features
##################################################################
def add_features_0(df):
    df["num_photos"] = df["photos"].apply(len)
    df["num_features"] = df["features"].apply(len)
    #fmt = lambda feat: [s.replace("\u00a0", "").strip().lower().replace(" ", "_") for s in feat]  # format features
    fmt = lambda x: " ".join(["_".join(i.split(" ")) for i in x])
    df["features"] = df["features"].apply(fmt)
    df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
    df["created"] = pd.to_datetime(df["created"])
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    df["created_hour"] = df["created"].dt.hour
    
    #df['price'] = df['price'].apply(np.log)
    
    df = df.fillna(-1).replace(np.inf, -1)
    
    return df

def add_priceoverbedbath (df):
    df = add_features_0(df)
    df["price_over_bedbath"] = df['price'] / (df['bathrooms'] + df['bedrooms'] + 1)
    df["price_over_bedbath"] = [ np.min([1000000, x]) for x in df['price_over_bedbath'] ]
    return df

def add_priceoverbedbath_priceoverbed (df):
    df = add_priceoverbedbath(df)
    df["price_over_bed"] = df['price'] / (df['bedrooms'] + 1)
    df["price_over_bed"] = [ np.min([1000000, x]) for x in df['price_over_bed'] ]
    return df

def add_priceoverbedbath_priceoverbed_priceoverbath (df):
    df = add_priceoverbedbath_priceoverbed(df)
    df["price_over_bath"] = df['price'] / (df['bathrooms'] + 1)
    df["price_over_bath"] = [ np.min([1000000, x]) for x in df['price_over_bath'] ]
    return df


    

##################################################################
# Handle high cardinality categorical features
##################################################################
def hcc_encode(train_df, test_df, variable, target, prior_prob, k, f=1, g=1, r_k=None, update_df=None):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    hcc_name = "_".join(["hcc", variable, target])

    grouped = train_df.groupby(variable)[target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    df = test_df[[variable]].join(grouped, on=variable, how="left")[hcc_name].fillna(prior_prob)
    if r_k: 
        df *= np.random.uniform(1 - r_k, 1 + r_k, len(test_df))     # Add uniform noise. Not mentioned in original paper

    if update_df is None: 
        update_df = test_df
    if hcc_name not in update_df.columns: 
        update_df[hcc_name] = np.nan
    update_df.update(df)
    return

def designate_single_observations(df1, df2, column):
    ps = df1[column].append(df2[column])
    grouped = ps.groupby(ps).size().to_frame().rename(columns={0: "size"})
    df1.loc[df1.join(grouped, on=column, how="left")["size"] <= 1, column] = -1
    df2.loc[df2.join(grouped, on=column, how="left")["size"] <= 1, column] = -1
    return df1, df2



##################################################################
# Data processing
##################################################################

def rename_photo_features(df):
    df['photos_size'] = df['size']
    df['photos_minval'] = df['mini']
    df['photos_maxval'] = df['maxi']
    df['photos_mean'] = df['mean']
    df['photos_std'] = df['std']
    df['photos_median'] = df['median']
    df['photos_percentile25'] = df['percent25']       
    df['photos_percentile75'] = df['percent75']
    for c in ['size', 'mini', 'maxi', 'mean', 'std', 'median', 'percent25', 'percent75']:
        del df[c]
    return df

def preprocess (train_df, test_df):
    # add features
    train_df = add_priceoverbedbath_priceoverbed_priceoverbath(train_df)
    test_df = add_priceoverbedbath_priceoverbed_priceoverbath(test_df)
    
    # add photos features
    train_photo_feat_file = data_path + '/photo_features/train_photo_features_20170402_125215.csv'
    test_photo_feat_file = data_path + '/photo_features/test_photo_features_20170402_222303.csv'
    trpf = pd.read_csv(train_photo_feat_file)
    rename_photo_features(trpf)
    tepf = pd.read_csv(test_photo_feat_file)
    rename_photo_features(tepf)
    train_df.append(trpf)
    test_df.append(tepf)
    
    train_df = train_df.replace({"interest_level": {"low": 0, "medium": 1, "high": 2}})
    train_df = train_df.join(pd.get_dummies(train_df["interest_level"], prefix="pred").astype(int))
    prior_0, prior_1, prior_2 = train_df[["pred_0", "pred_1", "pred_2"]].mean()

    # handle signe observations
    for col in ('building_id', 'manager_id', 'display_address'):
        train_df, test_df = designate_single_observations(train_df, test_df, col)

    np.random.seed(123)
    # High-Cardinality Categorical encoding
    skf = StratifiedKFold(5, random_state = 123)
    attributes = product(("building_id", "manager_id"), zip(("pred_1", "pred_2"), (prior_1, prior_2)))
    for variable, (target, prior) in attributes:
        hcc_encode(train_df, test_df, variable, target, prior, k=5, r_k=None)
        for train, test in skf.split(np.zeros(len(train_df)), train_df['interest_level']):
            hcc_encode(train_df.iloc[train], train_df.iloc[test], variable, target, prior, k=5, r_k=0.01, update_df=train_df)

    for c in ['pred_0', 'pred_1', 'pred_2']:
        del train_df[c]

    return train_df, test_df


def create_train_test_sets (train_df, test_df):
    cat_feats = ["display_address", "manager_id", "building_id", "street_address"]
    # Creating the label encoders
    for f in cat_feats:
        if train_df[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f]) + list(test_df[f]))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
    
    features_to_use = list(train_df.keys().values)
    features_to_use = [str(x) for x in features_to_use]
    remove_list = ['interest_level',
                   'created',
                   'photos',
                   'features',
                   'description',
#                   'display_address', 
                   ]
    for c in remove_list:
        features_to_use.remove(c)
                                   
    tfidf = CountVectorizer(stop_words='english', max_features=200)
    feat_sparse_train = tfidf.fit_transform(train_df["features"])
    feat_sparse_test = tfidf.transform(test_df["features"])
    # compressed sparse row matrices
    X_train = sparse.hstack([train_df[features_to_use], feat_sparse_train]).tocsr()
    X_test = sparse.hstack([test_df[features_to_use], feat_sparse_test]).tocsr()    

    y_train = np.array(train_df['interest_level'])
    
    return X_train, y_train, X_test
    


##################################################################
# Training function
##################################################################
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=2000, verbose=True):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.02
    param['max_depth'] = 4
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


def predictXGB(train_df, test_df, train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=2000, verbose=True):
    pred_test_y, model = runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=2000, verbose=True)
    
    # Linear regression : listing_id = f(created)
    x = train_df['created'].astype(int).values[:,np.newaxis]
    y = train_df['listing_id'].values
    lr = LinearRegression()
    lr.fit(x,y)
    
    cr = test_df['created'].astype(int).values[:,np.newaxis]
    lid = test_df['listing_id'].values
    d = lid - lr.predict(cr)
    
    preds = pd.DataFrame(pred_test_y)
    preds["d"] = d
    preds.loc[pr['d'] > 250000, 0] = 1
    preds.loc[pr['d'] > 250000, 1] = 0
    preds.loc[pr['d'] > 250000, 2] = 0
    del preds["d"]
    pred_test_y = preds.values

    return pred_test_y, model


##################################################################
# Cross validation 
##################################################################
def crossval (train_df, test_df, X_train, y_train, Nfolds):
    cv_scores = []
    kf = KFold(n_splits=Nfolds, shuffle=True, random_state=seed)
    for dev_index, val_index in kf.split(range(X_train.shape[0])):
        dev_X, val_X = X_train[dev_index,:], X_train[val_index,:]
        dev_y, val_y = y_train[dev_index], y_train[val_index]
        dev_X_df = train_df.ix[dev_index]
        val_X_df = train_df.ix[val_index]
        preds, model = predictXGB(dev_X_df, val_X_df, dev_X, dev_y, val_X, val_y)
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
    return cv_scores


##################################################################
# Output files saving
##################################################################
def create_output_files (train_df, test_df, X_train, y_train, X_test, test_listing_id, cv_scores = None, num_rounds=400):
    # make predictions
    preds, model = predictXGB(train_df, test_df, X_train, y_train, X_test, num_rounds=num_rounds)
    out_df = pd.DataFrame(preds)
    out_df.columns = ["low", "medium", "high"]
    out_df["listing_id"] = test_df['listing_id'].values
    
    # extension for saved files
    date = mp.strdate()
    extension = date
    if DO_CV == True:
        if cv_scores is not None:
            extension += '_CV_' + str(round(np.mean(cv_scores), 5))

    # save files
    out_df.to_csv(predictions_path + "xgb3" + extension + ".csv", index=False)
    pickle.dump(model, open(models_path + "xgb3" + extension + ".dat", "wb"))




##################################################################
# MAIN
##################################################################


# Load the data
if LOAD_DATA == True:
    train_df = pd.read_json(open(data_path + "train.json", "r")).sort_values(by="listing_id")
    test_df = pd.read_json(open(data_path + "test.json", "r")).sort_values(by="listing_id")
    
    train_df, test_df = preprocess(train_df, test_df)
    X_train, y_train, X_test = create_train_test_sets(train_df, test_df)


# Cross validation
if DO_CV == True:
    cv_scores = crossval(train_df, test_df, X_train, y_train, 5)
    #cv = ShuffleSplit(n_splits=cv_n_splits, test_size=cv_test_size, random_state=seed)
    #cv_score = cross_val_score(clf, X_train, y_train, cv=cv)
    print('CV score: log_loss = ' + str(np.mean(cv_scores)))
    if (valid_size > 0):
        preds, model = predictXGB(train_df, test_df, X_train, y_train, X_val, num_rounds=2000)
        val_score = log_loss(y_val, preds)
        print('Validation set: log_loss = ' + str(val_score))



# creating output file
if CREATE_SUBMISSION_FILE == True:
    create_output_files(train_df, test_df, X_train, y_train, X_test, test_listing_id, cv_scores, num_rounds=2000)
    