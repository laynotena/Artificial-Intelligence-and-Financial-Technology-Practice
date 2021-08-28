# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:02:05 2019

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
data = np.load('data.npy')
plt.plot(data)

tc = 1158
β = 0.5
Φ = 3
ω = 11

def lppl(t,tc,β,Φ,ω,A,B,C):
    return np.exp(A+B*((tc-t)**β)*(1+C*np.cos(ω*np.log(tc-t)+Φ)))

def Energy(t,tc,β,Φ,ω,A,B,C):
    return np.sum(abs(data[0:tc]-lppl(t,tc,β,Φ,ω,A,B,C)))

def linear(tc,β,Φ,ω):
    n = tc
    A = np.zeros((n,3))
    #ABC
    for i in range(n):
        A[i,0] = 1
        #A
        A[i,1] = (tc-i)**β
        #B
        A[i,2] = ((tc-i)**β)*np.cos(ω*np.log(tc-i)+Φ)
        #C
    x = np.linalg.lstsq(A, np.log(data[0:tc]))[0]
    print(x)
    A = x[0]
    B = x[1]
    C = x[2]/x[1]
    print('A=',A,'B=',B,'C=',C)
    return A,B,C

def generation(A,B,C):
    pop = np.random.randint(0,2,(10000,34))
    fit = np.zeros((10000,1))
    for generation in range(10):
        #print(generation)
        for i in range(10000):
            gene = pop[i,:]
            
            tc = np.sum(2**np.array(range(4))*gene[:4])+1151
            #[1151:1166]
            β  = np.sum(2**np.array(range(10))*gene[4:14])/1023
            # 0<β<1
            ω  = (np.sum(2**np.array(range(10))*gene[14:24])/1023)*18+2
            # 2<ω<20
            Φ  = (np.sum(2**np.array(range(10))*gene[24:])/1023)*(2*np.pi)
            # 0<Φ<2π        
            fit[i] = Energy(np.arange(tc),tc,β,Φ,ω,A,B,C)
        sortf = np.argsort(fit[:,0])
        pop = pop[sortf,:]
        for i in range(100,10000):
            fid = np.random.randint(0,100)
            mid = np.random.randint(0,100) 
            while mid ==fid:
                mid = np.random.randint(0,100)
            mask = np.random.randint(0,2,(1,34))
            son = pop[mid,:]
            father = pop[fid,:]
            son[mask[0,:]==1] = father[mask[0,:]==1]
            pop[i,:] = son
            
        for i in range(1000):
            m = np.random.randint(0,10000)
            n = np.random.randint(0,34)
            pop[m,n] = 1-pop[m,n]
        
    for i in range(10000):
        gene = pop[i,:]
        tc = np.sum(2**np.array(range(4))*gene[:4])+1151
        β  = np.sum(2**np.array(range(10))*gene[4:14])/1023
        ω  = (np.sum(2**np.array(range(10))*gene[14:24])/1023)*18+2
        Φ  = (np.sum(2**np.array(range(10))*gene[24:])/1023)*(2*np.pi)
        fit[i] = Energy(np.arange(tc),tc,β,Φ,ω,A,B,C)
    
    sortf = np.argsort(fit[:,0])
    pop = pop[sortf,:]
    gene = pop[0,:]
    tc = np.sum(2**np.array(range(4))*gene[:4])+1151
    β  = np.sum(2**np.array(range(10))*gene[4:14])/1023
    ω  = (np.sum(2**np.array(range(10))*gene[14:24])/1023)*18+2
    Φ  = (np.sum(2**np.array(range(10))*gene[24:])/1023)*(2*np.pi)
    print('tc=',tc,'β=',β,'Φ=',Φ,'ω=',ω, )
    return tc,β,Φ,ω
    
for i in range(100):
    print('第',i+1,'輪')
    A, B, C = linear(tc,β,Φ,ω)
    tc,β,Φ,ω = generation(A,B,C)
    if (i+1)%10==0:
        plt.plot(data)
        plt.plot(lppl(np.arange(tc),tc,β,Φ,ω,A,B,C))
        plt.show()