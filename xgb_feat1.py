#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 11:40:23 2017

@author: antoinemovschin

Recherche de features d'intérêt

"""



from xgb1 import add_features, create_train_test_sets, crossval


def add_price2 (df):
    df = add_features(df)
    df["price2"] = df["price"].apply(lambda x: x**2)
    return df

def add_bed2 (df):
    df = add_features(df)
    df["bedrooms2"] = df["bedrooms"].apply(lambda x: x**2)
    return df

def add_bath2 (df):
    df = add_features(df)
    df["bathrooms2"] = df["bathrooms"].apply(lambda x: x**2)
    return df

def add_priceoverbed (df):
    df = add_features(df)
    df["price_over_bed"] = df['price'] / df['bedrooms']
    df["price_over_bed"] = [ np.min([1000000, x]) for x in df['price_over_bed'] ]
    return df

def add_priceoverbath (df):
    df = add_features(df)
    df["price_over_bath"] = df['price'] / df['bathrooms']
    df["price_over_bath"] = [ np.min([1000000, x]) for x in df['price_over_bath'] ]
    return df

def add_priceoverbedbath (df):
    df = add_features(df)
    df["price_over_bedbath"] = df['price'] / (df['bathrooms'] + df['bedrooms'])
    df["price_over_bedbath"] = [ np.min([1000000, x]) for x in df['price_over_bedbath'] ]
    return df



adders = [
        add_price2,
        add_bed2,
        add_bath2,
        add_priceoverbed,
        add_priceoverbath,
        add_priceoverbedbath,
        ]
added = [
        "price2",
        "bedrooms2",
        "bathrooms2",
        "price_over_bed",
        "price_over_bath",
        "price_over_bedbath",
        ]

res_cv = {}
for n in range(len(adders)):
    fct = adders[n]
    feat = added[n]
    print(fct.func_name)
    X_train, y_train, X_test = create_train_test_sets(preprocess = fct, features_to_add = [feat])
    cv_scores = crossval(X_train, y_train, 5)
    print('CV score: log_loss = ' + str(np.mean(cv_scores)))
    res_cv[fct.func_name] = np.mean(cv_scores)

