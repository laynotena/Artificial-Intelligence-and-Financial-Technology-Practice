# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 21:00:32 2019

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt

call = {10800:189,10900:120, 11000:67, 11100:32, 11200:13}
put = {10800:63,10900:93, 11000:139, 11100:207, 11200:290}
x = np.arange(10700,11301)

def callr(K):
    global x 
    global call
    return np.maximum(x-K ,0)-call[K]

def putr(K):
    global x
    global put
    return np.maximum(K-x, 0)-put[K]
#%%
#bull spread
    
K_price = [10900, 11000, 11100]

def bull_spread(price_list):
    
    result = []
    
    for i in range(len(price_list)):
        for a in range(len(price_list)):
            if price_list[i] < price_list[a]:
                result.append([price_list[i],price_list[a]])
    return result

bull_spread_K = bull_spread(K_price)

for i in range(len(bull_spread_K)):
    y1 = callr(bull_spread_K[i][0])
    y2 = -callr(bull_spread_K[i][1])
    y3 = y1 +y2
    y4 = putr(bull_spread_K[i][0])
    y5 = -putr(bull_spread_K[i][1])
    y6 = y4 +y5
    if i==0:
        plt.plot(x, y3, 'r', label=('call',bull_spread_K[i][0],bull_spread_K[i][1]))
        plt.plot(x, y6, 'c', label=('put',bull_spread_K[i][0],bull_spread_K[i][1]))
    if i==1:
        plt.plot(x, y3, 'b', label=('call',bull_spread_K[i][0],bull_spread_K[i][1]))
        plt.plot(x, y6, 'm', label=('put',bull_spread_K[i][0],bull_spread_K[i][1]))
    if i==2:
        plt.plot(x, y3, 'g', label=('call',bull_spread_K[i][0],bull_spread_K[i][1]))
        plt.plot(x, y6, 'y', label=('put',bull_spread_K[i][0],bull_spread_K[i][1]))
    plt.plot ([x[0], x[-1]], [0,0],'--k')
    plt.legend()
    plt.show
    
#%%
#straddle&strangle

y7 = callr(11000)
y8 = putr(11100)
y9 = y7 +y8
y10 = -callr(11000)
y11 = -putr(11100)
y12 = y10 +y11
y13 = callr(10800)
y14 = putr(11200)
y15 = y13 +y14
y16 = -callr(10800)
y17 = -putr(11200)
y18 = y16 +y17 
plt.plot(x, y9, 'r', label='straddle')
plt.plot(x, y12, 'r')
plt.plot(x, y15, 'g', label='strangle')
plt.plot(x, y18, 'g')
plt.plot ([x[0], x[-1]], [0,0],'--k')
plt.legend()
plt.show

#%%
#butterfly spread
y19 = callr(11200)
y20 = callr(10800)
y21 = -callr(11000) * 2
y22 = y19 +y20 +y21
y23 = -callr(11100)
y24 = -callr(10900)
y25 = callr(11000) * 2
y26 = y23 +y24 +y25
plt.plot(x, y22, 'y',label=('butterfly spread : long call 10800*1 + 11200*1 , short call 11000*2'))
plt.plot(x, y26, 'c',label=('butterfly spread : long call 11000*2 , short call 111000*1 + 10900*1'))
plt.plot ([x[0], x[-1]], [0,0],'--k')
plt.legend()
plt.show()