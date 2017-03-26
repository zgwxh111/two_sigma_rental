#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:33:37 2017

@author: antoinemovschin
"""

from PIL import Image
import requests
from StringIO import StringIO
import pandas as pd


def get_photo_features(df):
    features = {}
    features['size'] = []
    features['mini'] = []
    features['maxi'] = []
    features['mean'] = []
    features['std'] = []
    features['median'] = []
    features['percent25'] = []
    features['percent75'] = []
    for n in range(df.shape[0]):
        if n % 100 == 0:
            print ('processing photos ... ' + str(n) + ' / ' + str(df.shape[0]))
        # list of photo URLs
        photos = df['photos'].values[n]
        # if there are no photos, all values are set to 0
        if len(photos) == 0:
            for k in features.keys():
                features[k].append(0)
            continue
          
        size = []
        mini = []
        maxi = []
        mean = []
        std = []
        median = []
        percent25 = []
        percent75 = []        
        for p in range(len(photos)):
            url = photos[p]
            # get image from URL
            response = requests.get(url)
            image = Image.open(StringIO(response.content))    
            # convert to grayscale
            image = image.convert(mode = "L")
            # convert to numpy array
            img = np.array(image.getdata())
            
            # size
            size.append(len(img))
            # min value
            mini.append(min(img))
            # max value
            maxi.append(max(img))
            # mean
            mean.append(np.mean(img))
            # standard deviation
            std.append(np.std(img))
            # median 
            median.append(np.median(img))
            # 1st quartile
            percent25.append(np.percentile(img, q=25))
            # 3rd quartile
            percent75.append(np.percentile(img, q=75))
            
            #plt.imshow(img.reshape(np.flipud(image.size)))
        
        features['size'].append(np.mean(size))
        features['mini'].append(np.mean(mini))
        features['maxi'].append(np.mean(maxi))
        features['mean'].append(np.mean(mean))
        features['std'].append(np.mean(std))
        features['median'].append(np.mean(median))
        features['percent25'].append(np.mean(percent25))
        features['percent75'].append(np.mean(percent75))
    
    return features


def add_photo_features(df):
    features = get_photo_features(df)    
    df['photos_size'] = features['size']
    df['photos_minval'] = features['mini']
    df['photos_maxval'] = features['maxi']
    df['photos_mean'] = features['mean']
    df['photos_std'] = features['std']
    df['photos_median'] = features['median']
    df['photos_percentile25'] = features['percent25']       
    df['photos_percentile75'] = features['percent75']

    return df



if __name__ == '__main__':
    directory = '../input/photo_features/'
    date = mp.strdate(format = '_YYYYMMDD')

    train_df = pd.read_json(open(data_path + "train.json", "r"))
    train_features = get_photo_features(train_df)
    train_features.to_csv(directory + "train_photo_features" + date + ".csv", index=False)

    test_df = pd.read_json(open(data_path + "test.json", "r"))
    test_features = get_photo_features(train_df)
    test_features.to_csv(directory + "test_photo_features" + date + ".csv", index=False)

        

