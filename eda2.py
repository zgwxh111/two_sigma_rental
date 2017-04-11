#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:55:27 2017

@author: antoinemovschin
"""

import numpy as np # linear algebra
import matplotlib.pyplot as plt
from scipy.stats import norm





# Linear regression : listing_id = f(created)
groups = train_df.groupby('interest_level')
x = train_df['created'].astype(int).values[:,np.newaxis]
y = train_df['listing_id'].values
lr = LinearRegression()
lr.fit(x,y)
for name, group in groups:
    plt.plot(group["created"].astype(int), group["listing_id"], 'o', label=name)
plt.plot(x, lr.predict(x), '-x', label='lr')
plt.legend()
plt.xlabel('created')
plt.ylabel('listing_id')
plt.grid(True)

b = lr.lr.intercept_
a = lr.coef_[0]
cr0 = train_df.loc[train_df['interest_level']==0, 'created'].astype(int).values[:,np.newaxis]
cr1 = train_df.loc[train_df['interest_level']==1, 'created'].astype(int).values[:,np.newaxis]
cr2 = train_df.loc[train_df['interest_level']==2, 'created'].astype(int).values[:,np.newaxis]
lid0 = train_df.loc[train_df['interest_level']==0, 'listing_id'].values
lid1 = train_df.loc[train_df['interest_level']==1, 'listing_id'].values
lid2 = train_df.loc[train_df['interest_level']==2, 'listing_id'].values

d0 = lid0 - lr.predict(cr0)
d1 = lid1 - lr.predict(cr1)
d2 = lid2 - lr.predict(cr2)

for (cr, d, n) in [(cr0, d0, 0), (cr1, d1, 1), (cr2, d2, 2)]:
    plt.plot(cr, d, 'o', label=str(n))
    plt.legend()
bins = 200
plt.figure()
for (cr, d, n) in [(cr0, d0, 0), (cr1, d1, 1), (cr2, d2, 2)]:
    plt.hist(d, bins=bins, alpha=.5, label = str(n))
    plt.legend()
    plt.axis([-20000, 20000, 0, 20000])
bins = 20
plt.figure()
for (cr, d, n) in [(cr0, d0, 0), (cr1, d1, 1), (cr2, d2, 2)]:
    plt.hist(d, bins=bins, alpha=.5, label = str(n))
    plt.legend()
    plt.axis([-20000, 800000, 0, 100])
    

plt.figure()
bins = 200
plt.hist(d1, bins=bins)
x = np.linspace(-25000, 25000, 50000)
plt.plot(x, 750 / .00007 * norm.pdf(x, loc = np.mean(d1), scale = np.std(d1)/2))
plt.axis([-25000, 25000, 0, 1600])

