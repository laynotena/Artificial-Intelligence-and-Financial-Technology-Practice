# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:52:16 2019

@author: Jessica Chiu
"""

import numpy as np
from PIL import Image
from scipy import signal

I = Image.open('1223.jpg')
data = np.asarray(I)
W,H = I.size

#原始圖片
data = data.astype('uint8')
I2 = Image.fromarray(data,'RGB')
I2.show()

#%%
#1.雜訊
data2 =data.copy()
data = data.astype('float64')
noise = np.random.normal(0,10,(H,W,3))
data3 = data+noise
data3[data3>255] = 255
data3[data3<0] = 0
data2 = data3.astype('uint8')
I2 = Image.fromarray(data2,'RGB')
I2.show()

#%%
#2.把雜訊除掉
x,y = np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
d = np.sqrt(x*x+y*y)
sigma, mu =0.05,0.0
M = np.exp(-((d-mu)**2/(2.0*sigma**2)))
M = M/np.sum(M[:])
#R = data[:,:,0]
#G = data[:,:,1]
#B = data[:,:,2]
R = data2[:,:,0]
G = data2[:,:,1]
B = data2[:,:,2]
R2 = signal.convolve2d(R,M,boundary='symm', mode='same')
G2 = signal.convolve2d(G,M,boundary='symm', mode='same')
B2 = signal.convolve2d(B,M,boundary='symm', mode='same')
data2[:,:,0] = R2.astype('uint8')
data2[:,:,1] = G2.astype('uint8')
data2[:,:,2] = B2.astype('uint8')

I2 = Image.fromarray(data2,'RGB')
I2.show()

#%%
#3.高斯模糊
I = Image.open('1223.jpg')
data = np.asarray(I)
W,H = I.size
data = data.astype('uint8')

#高斯模糊
x,y = np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
d = np.sqrt(x*x+y*y)
sigma, mu =1.0,0.0
M = np.exp(-((d-mu)**2/(2.0*sigma**2)))
M = M/np.sum(M[:])
R = data[:,:,0]
G = data[:,:,1]
B = data[:,:,2]
R2 = signal.convolve2d(R,M,boundary='symm', mode='same')
G2 = signal.convolve2d(G,M,boundary='symm', mode='same')
B2 = signal.convolve2d(B,M,boundary='symm', mode='same')
data2 = data.copy()
data2[:,:,0] = R2.astype('uint8')
data2[:,:,1] = G2.astype('uint8')
data2[:,:,2] = B2.astype('uint8')

I2 = Image.fromarray(data2,'RGB')
I2.show()
#%%
#4.Sobel Edge Detection
data2 = data.copy()
R = data2[:,:,0].astype('float')
G = data2[:,:,1].astype('float')
B = data2[:,:,2].astype('float')
gray = ((R+G+B)/3).astype('uint8')
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_y =  np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
I_x = signal.convolve2d(gray,sobel_x,boundary='symm', mode='same')
I_y = signal.convolve2d(gray,sobel_y,boundary='symm', mode='same')
data2 = (I_x**2)+(I_y**2)
data2_1D = np.sort(data2.reshape((-1,1)))[:,0]
threshold = data2_1D[int(len(data2_1D)*0.9)]
data3 = data2.copy()
data3[data2<threshold] = 255
data3[data2>threshold] = 0
I2 = Image.fromarray(data3)
I2.show()