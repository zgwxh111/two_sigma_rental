# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 19:53:35 2017

@author: antoinemovschin
"""



import numpy as np # linear algebra
import matplotlib.pyplot as plt
import json
import sys
from collections import Counter


data_path = '../input'


LOAD_DATA = False


# fonction qui met un feature sous forme de liste. Permet du transtypage par 
# la fonction donnée dans le paramètre 'cast'.
# ex: aslist(train0, 'interest_level', cast=str) --> ['medium', 'medium', 'high' ...]
def aslist(dic, feat='listing_id', cast=None):
    if cast: 
        l = [cast(u) for u in dic[feat].values()]
    else:
        l = dic[feat].values()
    return l




if (LOAD_DATA == True):
    with open(data_path + '/train.json') as data_file:    
        train0 = json.load(data_file)
    with open(data_path + '/test.json') as data_file:    
        test0 = json.load(data_file)
   
# affichage des noms des différentes variables
[str(u) for u in train0.keys()]

# Variable 'interest_level': calcul du nombre d'occurrences
tril = aslist(train0, 'interest_level', str)
{level : sum([f == level for f in tril]) for level in set(tril)}
# or alternatively:
#Counter(tril)
ind_med = [i for i, x in enumerate(tril) if x =='medium']
ind_lo = [i for i, x in enumerate(tril) if x =='low']
ind_hi = [i for i, x in enumerate(tril) if x =='high']


# Variable 'bedrooms"
trbed = aslist(train0, 'bedrooms')
bedcount = {level : sum([f == level for f in trbed]) for level in set(trbed)}
bins=20
plt.hist([trbed[i] for i in ind_lo], bins=bins, alpha=0.5, label='low')
plt.hist([trbed[i] for i in ind_med], bins=bins, alpha=0.5, label='medium')
plt.hist([trbed[i] for i in ind_hi], bins=bins, alpha=0.5, label='high')
plt.legend()
plt.grid(True)
plt.xlabel('bedrooms')
plt.ylabel('count')
plt.show()




