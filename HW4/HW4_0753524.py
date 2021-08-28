# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:48:03 2019

@author: Jessica Chiu
"""
import numpy as np
import random
from sklearn import datasets
iris = datasets.load_iris()
data = iris['data']
target = iris['target']
from collections import Counter 

def kmeans(sample, K, maxiter):
    N = sample.shape[0]
    D = sample.shape[1]
    C = np.zeros((K,D))
    L = np.zeros((N,1))
    L1 = np.zeros((N,1))
    dist = np.zeros((N,K))
    #range(N)是範圍，隨機抽K個數字
    idx = random.sample(range(N),K)
    C = sample[idx,:]
    iter = 0
    while(iter<maxiter):
        for i in range(K):
            dist[:,i] = np.sum((sample-np.tile(C[i,:],(N,1)))**2,1)
            L1 = np.argmin(dist,1)
            if (iter>0 and np.array_equal(L,L1)):
                break
        L = L1
        for i in range(K):
            idx = np.nonzero(L==i)[0]
            if(len(idx)>0):
                C[i,:] = np.mean(sample[idx,:],0)
        iter+=1
        
    wicd = np.sum(np.sqrt(np.sum((sample-C[L,:])**2,1)))
    return C, L, wicd

#未標準化
kmeans(data, 3, 1000)
#standard score
GA = (data-np.tile(np.mean(data,0),(data.shape[0],1)))/np.tile(np.std(data,0),(data.shape[0],1))
kmeans(GA, 3, 1000)
#scaling
SC = (data-np.tile(np.min(data,0),(data.shape[0],1)))/(np.tile(np.max(data,0),(data.shape[0],1))-np.tile(np.min(data,0),(data.shape[0],1)))
kmeans(SC, 3, 1000)

#%%

def knn(test, train, target, k):
    N = train.shape[0]
    dist = np.sum((train-np.tile(test,(N,1)))**2,1)
    idx = sorted(range(len(dist)), key=lambda i:dist[i])[0:k]
    return Counter(target[idx]).most_common(1)[0][0]


#knn
for i in range(10):
    confusion_matrix = np.zeros((3,3))
    prediction = []
    for j in range(len(data)):
        test_data = data[j]
        train_data = np.vstack([data[0:j],data[j+1:len(data)]])
        prediction.append(knn(test_data,train_data, target, i+1))
    for t in range(len(data)):
        confusion_matrix[prediction[t], target[t]] +=1
    print(i+1)
    print(confusion_matrix)
    
