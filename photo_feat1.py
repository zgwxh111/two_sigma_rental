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
    nrows = df.shape[0]
    features = {}
    features['size'] = np.zeros((nrows,))
    features['mini'] = np.zeros((nrows,))
    features['maxi'] = np.zeros((nrows,))
    features['mean'] = np.zeros((nrows,))
    features['std'] = np.zeros((nrows,))
    features['median'] = np.zeros((nrows,))
    features['percent25'] = np.zeros((nrows,))
    features['percent75'] = np.zeros((nrows,))
    
    err_list = np.zeros((nrows,))
    err_list[:] = -1

    #for n in range(df.shape[0]):
    for n in range(64085, df.shape[0]):
        if n % 1000 == 0:
            # sauvegarde tous les 1000 annonces
            df_features = pd.DataFrame(features)
            directory = '../input/photo_features/'
            date = mp.strdate()
            df_features.to_csv(directory + "df_photo_features" + date + ".csv", index=False)
        if n % 100 == 0:
            print ('processing photos ... ' + str(n) + ' / ' + str(df.shape[0]))
        # list of photo URLs
        photos = df['photos'].values[n]
        nphotos = len(photos)
        # if there are no photos, all values are set to 0
        if nphotos == 0:
            for k in features.keys():
                features[k][n] = 0
            continue
        
        size = np.zeros((nphotos,))
        mini = np.zeros((nphotos,))
        maxi = np.zeros((nphotos,))
        mean = np.zeros((nphotos,))
        std = np.zeros((nphotos,))
        median = np.zeros((nphotos,))
        percent25 = np.zeros((nphotos,))
        percent75 = np.zeros((nphotos,))        
        for p in range(nphotos):
            url = photos[p]
            # get image from URL
            try:
                response = requests.get(url)
            except ConnectionError:
                image = None
                print ("error image " + str(p) + ", n = " + str(n))
                err_list[n] = p
            try:
                image = Image.open(StringIO(response.content))    
            except IOError:
                image = None
                print ("error image " + str(p) + ", n = " + str(n))
                err_list[n] = p
            
            if image == None:
                # on remet les caractéristiques de la dernière image valide
                # TODO: modifier (on peut faire mieux)
                # size
                size[p] = len(img)
                # min value
                mini[p] = min(img)
                # max value
                maxi[p] = max(img)
                # mean
                mean[p] = np.mean(img)
                # standard deviation
                std[p] = np.std(img)
                # median 
                median[p] = np.median(img)
                # 1st quartile
                percent25[p] = np.percentile(img, q=25)
                # 3rd quartile
                percent75[p] = np.percentile(img, q=75)
                
            else:
                # convert to grayscale
                image = image.convert(mode = "L")
                # convert to numpy array
                img = np.array(image.getdata())
                
                # size
                size[p] = len(img)
                # min value
                mini[p] = min(img)
                # max value
                maxi[p] = max(img)
                # mean
                mean[p] = np.mean(img)
                # standard deviation
                std[p] = np.std(img)
                # median 
                median[p] = np.median(img)
                # 1st quartile
                percent25[p] = np.percentile(img, q=25)
                # 3rd quartile
                percent75[p] = np.percentile(img, q=75)
                
                #plt.imshow(img.reshape(np.flipud(image.size)))
        
        features['size'][n] = np.mean(size)
        features['mini'][n] = np.mean(mini)
        features['maxi'][n] = np.mean(maxi)
        features['mean'][n] = np.mean(mean)
        features['std'][n] = np.mean(std)
        features['median'][n] = np.mean(median)
        features['percent25'][n] = np.mean(percent25)
        features['percent75'][n] = np.mean(percent75)
    
    return features, err_list


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

    train_df = pd.read_json(open(data_path + "train.json", "r"))
    train_ft = get_photo_features(train_df)
    train_features = pd.DataFrame(train_ft)
    date = mp.strdate()
    train_features.to_csv(directory + "train_photo_features" + date + ".csv", index=False)

    test_df = pd.read_json(open(data_path + "test.json", "r"))
    test_ft = get_photo_features(train_df)
    test_features = pd.DataFrame(test_ft)
    date = mp.strdate()
    test_features.to_csv(directory + "test_photo_features" + date + ".csv", index=False)


#trerr = [16193, 22480, 25711, 28352, 28801, 29541,
#         38009, 38336, 39480, 40362, 40373, 40812, 
#         40813, 40814, 40821, 40831, 40854, 43646, 
#         44277, 44611, 44816, 44819, 44822, 45170, 
#         45181, 45619, 45625, 45737, 45791, 46116, 
#         46219, 46958, 47064, 47290, 47353, 47436, 
#         48046, 48623, 48798, 48963, 49033, 49313]
