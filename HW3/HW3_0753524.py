# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:48:56 2019

@author: Jessica Chiu
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from  scipy.stats import norm

def MCsim(S,T,r,vol,N):
    dt = T/N
    global St
    St[0] = S
    for i in range(N):
        St[i+1] = St[i]*math.exp((r-0.5*vol*vol)*dt+np.random.normal()*vol*math.sqrt(dt))
    return St

def BLSprice(S,L,T,r,vol):
    d1 = (math.log(S/L)+(r+0.5*vol*vol)*T) / (vol*math.sqrt(T))
    d2 = d1-vol*math.sqrt(T)
    call = S*norm.cdf(d1) - L*math.exp(-r*T)*norm.cdf(d2)
    return call

S = 50
L = 40
T = 2
r = 0.08
vol = 0.2
N = 100
St = np.zeros((N+1))

M = 100000
call = 0
for i in range(M):
    Sa = MCsim(S,T,r,vol,N)
    if (Sa[-1]-L>0):
        call += Sa[-1]-L
print(call/M*math.exp(-r*T))
print(BLSprice(S,L,T,r,vol))


#%%

def BTcall(S,T,r0,vol,N,L):
    dt = T/N
    u = math.exp(vol*math.sqrt(dt))
    d = math.exp(-vol*math.sqrt(dt))
    p = (math.exp(r0*dt)-d) / (u-d)
    priceT = np.zeros((N+1,N+1))
    priceT[0][0] = S
    for c in range(N):
        priceT[0][c+1] = priceT[0][c] * u
        for r in range(c+1):
            priceT[r+1][c+1] = priceT[r][c]*d
    probT = np.zeros((N+1,N+1))
    probT[0][0] = 1
    for c in range(N):
        for r in range(c+1):
            probT[r][c+1] += probT[r][c]*p 
            probT[r+1][c+1] += probT[r][c]*(1-p) 
    call = 0
    for r in range(N+1):
        if (priceT[r][N] >= L) :
            call += (priceT[r][N]-L)*probT[r][N]
    return call*math.exp(-r0*T)

S = 50
L = 40
T = 2
r = 0.08
vol = 0.2
N = 100
St = np.zeros((N+1))

print(BTcall(S,T,r,vol,N,L))
print(BLSprice(S,L,T,r,vol))

#%%

def BisectionBLS(S,L,T,r,call,tol):
    left = 0.00000000000001
    right = 1
    while(right-left>tol):
        middle = (left+right)/2
        if((BLSprice(S,L,T,r,middle)-call)*(BLSprice(S,L,T,r,left)-call)<0):
            right = middle
        else:
            left = middle
    return (left+right)/2

L_list = [[10700,406], [10900,209], [11000,112], [11300,0.1], [11400,0.1]]
K_list = [10700, 10900, 11000, 11300,11400]
r_list = []

S = 10889.96
L = 10700
T = 7/365
r = 0.006
call = 406
vol = 0.02
N = 100
St = np.zeros((N+1))

call = BLSprice(S,L,T,r,vol)
print(BisectionBLS(S,L,T,r,call,0.00001))

for x,y in L_list:
    L = x
    call = y
    result = BisectionBLS(S,L,T,r,call,0.00001)
    r_list.append(result)

plt.plot(K_list, r_list, 'r')



#%%

#Sa = MCsim(S,T,r,vol,N)
#plt.plot(Sa)

print(call/M*math.exp(-r*T))
print(BTcall(S,T,r,vol,N,L))

