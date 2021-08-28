# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:49:01 2019

@author: User
"""

import numpy as np
import pandas as pd
import math

#%%
#entropy
def entropy(p1, n1):
    if (p1==0 and n1==0):
        return 1
    elif (p1==0):
        return 0
    elif (n1==0):
        return 0
    pp = p1 / (p1+n1)
    pn = n1 / (p1+n1)
    return -pp*math.log2(pp)-pn*math.log2(pn)

#Information Gain
def IG(p1, n1, p2, n2):
    num1 = p1+n1
    num2 = p2+n2
    num = num1+num2
    return entropy(p1+p2,n1+n2)-num1/num*entropy(p1,n1)-num2/num*entropy(p2,n2)
#%%

#讀取資料集
from sklearn import datasets
iris = datasets.load_iris()

#讓每次random的數字都一樣
np.random.seed(38)

#這樣對150的數字做排列的結果才會一樣
idx = np.random.permutation(150)

#取出feature和targets
features = iris.data[idx,:]
targets = iris.target[idx]
prediction = []
size = features.shape[0]//5
test_Y_set = []
test_Y_set = np.array(test_Y_set)
"""
Tree1=(0,1), Tree2=(1,2), Tree3=(0,2)
"""
   
for m in range(5):
    Tree1_data, Tree1_targets, Tree2_data, Tree2_targets, Tree3_data, Tree3_targets = [],[],[],[],[],[]   
    test_X = features[m*size:(m+1)*size]
    test_Y = targets[m*size:(m+1)*size]
    test_Y_set = np.hstack((test_Y_set,test_Y))
    train_X = np.vstack([features[0:m*size],features[(m+1)*size:]])
    train_Y = np.hstack([targets[0:m*size],targets[(m+1)*size:]])
   
    for n in range(len(train_Y)):   
        if train_Y[n]==0:
            Tree1_data.append(train_X[n])
            Tree1_targets.append(train_Y[n])
            Tree3_data.append(train_X[n])
            Tree3_targets.append(train_Y[n])
        elif train_Y[n]==1:
            Tree1_data.append(train_X[n])
            Tree1_targets.append(train_Y[n])
            Tree2_data.append(train_X[n])
            Tree2_targets.append(train_Y[n])
        elif train_Y[n]==2:
            Tree2_data.append(train_X[n])
            Tree2_targets.append(train_Y[n])
            Tree3_data.append(train_X[n])
            Tree3_targets.append(train_Y[n])
            
    Tree1_data = np.array(Tree1_data)
    Tree1_targets = np.array(Tree1_targets)
    Tree2_data = np.array(Tree2_data)
    Tree2_targets = np.array(Tree2_targets)
    Tree3_data = np.array(Tree3_data)
    Tree3_targets = np.array(Tree3_targets)
#%%  
    node = dict()    
    node['data'] = range(len(Tree1_targets))
    Tree1 = []
    Tree1.append(node)
    t= 0
    while(t<len(Tree1)):
        idx = Tree1[t]['data']
        if (sum(Tree1_targets[idx])==0):
            Tree1[t]['leaf']=1
            Tree1[t]['decision']=0
        elif(sum(Tree1_targets[idx])==len(idx)):
            Tree1[t]['leaf']=1
            Tree1[t]['decision']=1
        else:
            bestIG = 0
            
            for i in range(Tree1_data.shape[1]):
                pool = list(set(Tree1_data[idx,i]))
                pool.sort()
                
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 =[]
                    G2 =[]
                    
                    for k in idx:
                        if (Tree1_data[k,i] <= thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = IG(sum(Tree1_targets[G1]==1), sum(Tree1_targets[G1]==0), sum(Tree1_targets[G2]==1), sum(Tree1_targets[G2]==0))
                    if (thisIG > bestIG):
                        bestIG = thisIG
                        bestf = i
                        bestthres = thres
                        bestG1 = G1
                        bestG2 = G2
            if(bestIG>0):
                Tree1[t]['leaf'] = 0
                Tree1[t]['selectf'] = bestf
                Tree1[t]['threshold'] = bestthres
                Tree1[t]['child'] = [len(Tree1),len(Tree1)+1]
                node = dict()
                node['data'] = bestG1
                Tree1.append(node)
                node = dict()
                node['data'] = bestG2
                Tree1.append(node)
            else:
                Tree1[t]['leaf'] = 1
                if (sum(Tree1_targets[idx]==1)>sum(Tree1_targets[idx]==0)):
                    Tree1[t]['decision'] = 1
                else:
                    Tree1[t]['decision'] = 0
        t+=1
        
#%%

    
    node = dict()    
    node['data'] = range(len(Tree2_targets))
    Tree2 = []
    Tree2.append(node)
    t= 0
    while(t<len(Tree2)):
        idx = Tree2[t]['data']
        if (sum(Tree2_targets[idx])==len(idx)):
            Tree2[t]['leaf']=1
            Tree2[t]['decision']=1
        elif(sum(Tree2_targets[idx])==(len(idx)*2)):
            Tree2[t]['leaf']=1
            Tree2[t]['decision']=2
        else:
            bestIG = 0
            
            for i in range(Tree2_data.shape[1]):
                pool = list(set(Tree2_data[idx,i]))
                pool.sort()
                
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 =[]
                    G2 =[]
                    
                    for k in idx:
                        if (Tree2_data[k,i] <= thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = IG(sum(Tree2_targets[G1]==2), sum(Tree2_targets[G1]==1), sum(Tree2_targets[G2]==2), sum(Tree2_targets[G2]==1))
                    if (thisIG > bestIG):
                        bestIG = thisIG
                        bestf = i
                        bestthres = thres
                        bestG1 = G1
                        bestG2 = G2
            if(bestIG>0):
                Tree2[t]['leaf'] = 0
                Tree2[t]['selectf'] = bestf
                Tree2[t]['threshold'] = bestthres
                Tree2[t]['child'] = [len(Tree2),len(Tree2)+1]
                node = dict()
                node['data'] = bestG1
                Tree2.append(node)
                node = dict()
                node['data'] = bestG2
                Tree2.append(node)
            else:
                Tree2[t]['leaf'] = 1
                if (sum(Tree2_targets[idx]==2)>sum(Tree2_targets[idx]==1)):
                    Tree2[t]['decision'] = 2
                else:
                    Tree2[t]['decision'] = 1
        t+=1
    


    
    node = dict()    
    node['data'] = range(len(Tree3_targets))
    Tree3 = []
    Tree3.append(node)
    t= 0
    while(t<len(Tree3)):
        idx = Tree3[t]['data']
        if (sum(Tree3_targets[idx])==0):
            Tree3[t]['leaf']=1
            Tree3[t]['decision']=0
        elif(sum(Tree3_targets[idx])==(len(idx)*2)):
            Tree3[t]['leaf']=1
            Tree3[t]['decision']=2
        else:
            bestIG = 0
            
            for i in range(Tree3_data.shape[1]):
                pool = list(set(Tree3_data[idx,i]))
                pool.sort()
                
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 =[]
                    G2 =[]
                    
                    for k in idx:
                        if (Tree3_data[k,i] <= thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = IG(sum(Tree3_targets[G1]==2), sum(Tree3_targets[G1]==0), sum(Tree3_targets[G2]==2), sum(Tree3_targets[G2]==0))
                    if (thisIG > bestIG):
                        bestIG = thisIG
                        bestf = i
                        bestthres = thres
                        bestG1 = G1
                        bestG2 = G2
            if(bestIG>0):
                Tree3[t]['leaf'] = 0
                Tree3[t]['selectf'] = bestf
                Tree3[t]['threshold'] = bestthres
                Tree3[t]['child'] = [len(Tree3),len(Tree3)+1]
                node = dict()
                node['data'] = bestG1
                Tree3.append(node)
                node = dict()
                node['data'] = bestG2
                Tree3.append(node)
            else:
                Tree3[t]['leaf'] = 1
                if (sum(Tree3_targets[idx]==2)>sum(Tree3_targets[idx]==0)):
                    Tree3[t]['decision'] = 2
                else:
                    Tree3[t]['decision'] = 0
        t+=1
    
        
    for i in range(len(test_Y)):
        predict = []

        test_feature = test_X[i,:]
        now = 0
        while(Tree1[now]['leaf']==0):
            if(test_feature[Tree1[now]['selectf']] <= Tree1[now]['threshold']):
                now = Tree1[now]['child'][0]
            else:
                now = Tree1[now]['child'][1]
        a = Tree1[now]['decision']

        now = 0
        while(Tree2[now]['leaf']==0):
            if(test_feature[Tree2[now]['selectf']] <= Tree2[now]['threshold']):
                now = Tree2[now]['child'][0]
            else:
                now = Tree2[now]['child'][1]         
        b = Tree2[now]['decision']

        now = 0
        while(Tree3[now]['leaf']==0):
            if(test_feature[Tree3[now]['selectf']] <= Tree3[now]['threshold']):
                now = Tree3[now]['child'][0]
            else:
                now = Tree3[now]['child'][1]
        c = Tree3[now]['decision']
        
        predict.append([a,b,c])
        #print( a,b,c)
        count0, count1, count2 = 0,0,0
        
        for p in predict[0]:
            #print(p)
            if p == 0 :
                count0 = count0 +1
            elif p == 1 :
                count1 = count1 +1
            else:
                count2 = count2 +1    
        if (count0 >count1 and count0>count2):
            prediction.append(0)
        elif (count1 >count0 and count1>count2):
            prediction.append(1)
        else:
            prediction.append(2)
#%%
def accuracy(prediction1, label):
    count = 0
    for i in range(len(label)):
        if prediction1[i] == label[i]:
           count = count+1
        else:
            print(i, label[i], prediction1[i])
    return count/len(label)

accuracy(prediction, test_Y_set)


#%%
def confusion_matrix(prediction1, label):
    count00, count01, count02, count10, count11, count12, count20, count21, count22=0,0,0,0,0,0,0,0,0
    for i in range(len(label)):
        if (prediction1[i] ==label[i] ==0):
            count00 = count00+1
        elif (prediction1[i] ==label[i] ==1):
            count11 = count11+1
        elif (prediction1[i] ==label[i] ==2):
            count22 = count22+1
        elif (prediction1[i] ==0 and label[i] ==1):
            count01 = count01+1
        elif (prediction1[i] ==0 and label[i] ==2):
            count02 = count02+1
        elif (prediction1[i] ==1 and label[i] ==0):
            count10 = count10+1
        elif (prediction1[i] ==1 and label[i] ==2):
            count12 = count12+1
        elif (prediction1[i] ==2 and label[i] ==0):
            count20 = count20+1
        elif (prediction1[i] ==2 and label[i] ==1):
            count21 = count21+1
        count = [[count00,count10,count20],[count01,count11,count21],[count02,count12,count22]]
        matrix = pd.DataFrame(count, columns = ['0','1','2'])

    return matrix

confusion_matrix(prediction, test_Y_set)
    
    
    
    
            
            
            
            
            
            
            
            
            
            
            
            
            
