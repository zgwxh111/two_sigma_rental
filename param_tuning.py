#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:02:42 2017

@author: antoinemovschin
"""

from xgb3 import crossval

params={}
params['eta'] = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2] 
params['max_depth'] = [3,4,5]
params['min_child_weight'] = [1, 2, 3]
params['gamma'] = [0.1, 0.2, 0.5]
num_rounds = 4000


result=pd.DataFrame([])

for eta in params['eta']:
    for max_depth in params['max_depth']:
        for min_child_weight in params['min_child_weight']:
            for gamma in params['gamma']:
                param = {}
                param['objective'] = 'multi:softprob'
                param['eta'] = eta 
                param['max_depth'] = max_depth
                param['silent'] = 1
                param['num_class'] = 3
                param['eval_metric'] = "mlogloss"
                param['min_child_weight'] = min_child_weight
                param['subsample'] = 0.7
                param['colsample_bytree'] = 0.7
                param['gamma'] = gamma
                param['seed'] = 0
                
                res = pd.DataFrame()
                res['eta'] = [eta]
                res['max_depth'] = [max_depth]
                res['min_child_weight'] = [min_child_weight]
                res['gamma'] = [gamma]
                cv_scores, nrounds = crossval(train_df, test_df, X_train, y_train, 4, num_rounds=num_rounds, param = param)
                res['cv_score'] = [np.mean(cv_scores)]
                res['nrounds'] = [np.max(nrounds)]
                result = result.append(res)
                
                date = mp.strdate()
                result.to_csv('../output/results' + date + '.csv', index=False)

