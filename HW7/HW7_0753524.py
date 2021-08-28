# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']

trpn = trainface.shape[0]
trnn = trainnonface.shape[0]
tepn = testface.shape[0]
tenn = testnonface.shape[0]

fn = 0
ftable = []
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h<=19 and x+w*2<=19):
                    fn = fn + 1
                    ftable.append([0,y,x,h,w])
print(fn)                    
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w<=19):
                    fn = fn + 1
                    ftable.append([1,y,x,h,w])
print(fn)
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h<=19 and x+w*3<=19):
                    fn = fn + 1
                    ftable.append([2,y,x,h,w])
print(fn)
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w*2<=19):
                    fn = fn + 1
                    ftable.append([3,y,x,h,w])
print(fn)

def fe(sample,ftable,c):
    ftype = ftable[c][0]
    y = ftable[c][1]
    x = ftable[c][2]
    h = ftable[c][3]
    w = ftable[c][4]
    T = np.arange(361).reshape((19,19))
    if(ftype==0):
        idx1 = T[y:y+h,x:x+w].flatten()
        idx2 = T[y:y+h,x+w:x+w*2].flatten()
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)
    elif(ftype==1):
        idx1 = T[y:y+h,x:x+w].flatten()
        idx2 = T[y+h:y+h*2,x:x+w].flatten()
        output = np.sum(sample[:,idx2],axis=1)-np.sum(sample[:,idx1],axis=1)
    elif(ftype==2):
        idx1 = T[y:y+h,x:x+w].flatten()
        idx2 = T[y:y+h,x+w:x+w*2].flatten()
        idx3 = T[y:y+h,x+w*2:x+w*3].flatten()
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)+np.sum(sample[:,idx3],axis=1)
    else:
        idx1 = T[y:y+h,x:x+w].flatten()
        idx2 = T[y:y+h,x+w:x+w*2].flatten()
        idx3 = T[y+h:y+h*2,x:x+w].flatten()
        idx4 = T[y+h:y+h*2,x+w:x+w*2].flatten()
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)-np.sum(sample[:,idx3],axis=1)+np.sum(sample[:,idx4],axis=1)
    return output
        
#trpf = np.zeros((trpn,fn)) #2429X36648
#trnf = np.zeros((trnn,fn)) #4548X36648

tepf = np.zeros((tepn,fn)) #2429X36648
tenf = np.zeros((tenn,fn)) #4548X36648


#for c in range(fn):
#    trpf[:,c] = fe(trainface,ftable,c)
#    trnf[:,c] = fe(trainnonface,ftable,c)
    
for c in range(fn):
    tepf[:,c] = fe(testface,ftable,c)
    tenf[:,c] = fe(testnonface,ftable,c)
    
def WC(pw,nw,pf,nf):
    maxf = max(pf.max(),nf.max())
    minf = min(pf.min(),nf.min())
    theta = (maxf-minf)/10+minf
    error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
    polarity = 1
    if(error>0.5):
        polarity = 0
        error = 1 - error
    min_theta = theta
    min_error = error
    min_polarity = polarity
    for i in range(2,10):
        theta = (maxf-minf)*i/10+minf
        error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
        polarity = 1
        if(error>0.5):
            polarity = 0
            error = 1 - error
        if(error<min_error):
            min_theta = theta
            min_error = error
            min_polarity = polarity
    return min_error,min_theta,min_polarity
    
pw = np.ones((tepn,1))/tepn/2
nw = np.ones((tenn,1))/tenn/2
SC = []


for t in range(200): #用200個弱分類器
    weightsum = np.sum(pw)+np.sum(nw)
    pw = pw/weightsum
    nw = nw/weightsum
    best_error,best_theta,best_polarity = WC(pw,nw,tepf[:,0],tenf[:,0])
    best_feature = 0
    for i in range(1,fn):
        me,mt,mp = WC(pw,nw,tepf[:,i],tenf[:,i])
        if(me<best_error):
            best_error = me
            best_feature = i
            best_theta = mt
            best_polarity = mp
    beta = best_error/(1-best_error)
    if(best_polarity == 1):
        pw[tepf[:,best_feature]>=best_theta]*=beta
        nw[tenf[:,best_feature]<best_theta]*=beta
    else:
        pw[tepf[:,best_feature]<best_theta]*=beta
        nw[tenf[:,best_feature]>=best_theta]*=beta
    alpha = np.log10(1/beta)
    SC.append([best_feature,best_theta,best_polarity,alpha])
    print(t)
    print(best_feature)
    
np.save('sc_te.npy',SC)

teps = np.zeros((tepn,1))
tens = np.zeros((tenn,1))
alpha_sum = 0
times = [1, 3, 5, 20, 100, 200]



for i in range(200):
    feature = int(SC[i][0])
    theta = SC[i][1]
    polarity = SC[i][2]
    alpha = SC[i][3]
    alpha_sum += alpha
    if(polarity==1):
        teps[tepf[:,feature]>=theta] += alpha
        tens[tenf[:,feature]>=theta] += alpha
    else:
        teps[tepf[:,feature]<theta] += alpha
        tens[tenf[:,feature]<theta] += alpha
    
    
teps /= alpha_sum
tens /= alpha_sum

x = []
y = []
for j in range(1000):
    threshold = j/1000
    x.append(np.sum(tens>=threshold)/tenn)
    y.append(np.sum(teps>=threshold)/tepn)

plt.plot(x,y)

#%%

I = Image.open('3.jpg')

width = 900
ratio = float(width)/I.size[0]
height = int(I.size[1]*ratio)
I = I.resize( (width, height), Image.BILINEAR )
print(I.size)

black = I.convert('L')
data = np.array(black)


I2 = Image.fromarray(data,'L')
I2.show()

num_w = width-19+1
num_h = height-19+1

candidates = np.zeros((num_w*num_h,361))
pos = []
count = 0
for i in range(num_h):
    for j in range(num_w):
        pos.append([i,j])
        candidates[count] = data[i:(i+19),j:(j+19)].flatten()
        count+=1

can_features = np.zeros((len(candidates),100))        
        
for c in range(100):
    feature = int(SC[c][0])
    can_features[:,c] = fe(candidates,ftable,feature)
                
can_points = np.zeros((len(candidates),1))
alpha_sum=0
for i in range(100):
    feature = int(SC[i][0])
    theta = SC[i][1]
    polarity = SC[i][2]
    alpha = SC[i][3]
    alpha_sum = alpha_sum+alpha       
    if polarity==1:
        can_points[can_features[:,i]>=theta]=can_points[can_features[:,i]>=theta]+alpha
    else:
        can_points[can_features[:,i]<theta]=can_points[can_features[:,i]<theta]+alpha
can_points = can_points/alpha_sum

where = np.where(can_points>0.52)[0]

data2 = np.array(I)
for i in where:
    x = pos[i][0]
    y = pos[i][1]
    data2[x,y:(y+19)] = [255,0,0]
    data2[(x+19),y:(y+19)] = [255,0,0]
    data2[x:(x+19),y] = [255,0,0]
    data2[x:(x+19),y+19] = [255,0,0]

I2 = Image.fromarray(data2,'RGB')
I2.show()

#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('TXF20112015.csv',sep=',',header=None)
TAIEX = df.values
tradeday = list(set(TAIEX[:,0]//10000))
tradeday.sort()

def show(profit):
    profit = np.array(profit)
    profit2 = np.cumsum(profit)
    global tradeday
    ans0 = len(tradeday) #總進場次數
    ans1 = profit2[-1] #總損益點數
    ans2 = np.sum(profit>0)/len(profit) #勝率
    ans3 = np.mean(profit[profit>0])#賺錢時獲利點數
    ans4 = np.mean(profit[profit<=0])#輸錢時損失點數
    print('entry times:',ans0,'\ntotal:',ans1,'\nwin ratio:',ans2,'\nwin avg:',ans3,'\nlose avg:',ans4,'\nprofit distribution:')
    plt.hist(profit,bins=100)
    plt.show()
    print('accumulated profit:')
    plt.plot(profit2)
    plt.show()
    
#strategy 0.0
profit = np.zeros((len(tradeday),1))
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    profit[i] = TAIEX[idx[-1],1] - TAIEX[idx[0],2]
print('************ strategy 0.0 ************')
show(profit)

#strategy0.1
profit = np.zeros((len(tradeday),1))
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    profit[i] = TAIEX[idx[0],2] - TAIEX[idx[-1],1]
print('************ strategy 0.1 ************')
show(profit)

#strategy1.0 
profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2] #開盤價買入
    idx2 = np.nonzero(TAIEX[idx,4]<=p1-30)[0] #idx是屬於今天的index，最低價拿出來 ＃idx2是一天的300分鐘內低於30點的分鐘
    if(len(idx2)==0):
        p2 = TAIEX[idx[-1],1] #沒有的話用平倉價買
    else:
        p2 = TAIEX[idx[idx2[0]],1] 
    profit.append(p2-p1)
print('************ strategy 1.0 ************')
show(profit)

#strategy1.1
profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2] 
    idx2 = np.nonzero(TAIEX[idx,4]<=p1-30)[0] 
    if(len(idx2)==0):
        p2 = TAIEX[idx[-1],1] 
    else:
        p2 = TAIEX[idx[idx2[0]],1] 
    profit.append(p1-p2)
print('************ strategy 1.1 ************')
show(profit)

#strategy2.0
profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2]
    idx2 = np.nonzero(TAIEX[idx,4]<=p1-30)[0] #停損
    idx3 = np.nonzero(TAIEX[idx,3]>=p1+30)[0] #停利
    if(len(idx2)==0 and len(idx3)==0):
        p2 = TAIEX[idx[-1],1]
    elif(len(idx3)==0): #沒有停利點只有停損點
        p2 = TAIEX[idx[idx2[0]],1] 
    elif(len(idx2)==0): #沒有停損點只有停利點
        p2 = TAIEX[idx[idx3[0]],1]
    #都有的話就比早，看停損停利誰比較早發生
    elif(idx2[0]<idx3[0]):
        p2 = TAIEX[idx[idx2[0]],1]
    else: 
        p2 = TAIEX[idx[idx3[0]],1] 
    profit.append(p2-p1)
print('************ strategy 2.0 ************')
show(profit)
#strategy2.1
profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2]
    idx2 = np.nonzero(TAIEX[idx,4]>=p1+30)[0] #停損
    idx3 = np.nonzero(TAIEX[idx,3]<=p1-30)[0] #停利
    if(len(idx2)==0 and len(idx3)==0):
        p2 = TAIEX[idx[-1],1]
    elif(len(idx3)==0): #沒有停利點只有停損點
        p2 = TAIEX[idx[idx2[0]],1] 
    elif(len(idx2)==0): #沒有停損點只有停利點
        p2 = TAIEX[idx[idx3[0]],1]
    #都有的話就比早，看停損停利誰比較早發生
    elif(idx2[0]<idx3[0]):
        p2 = TAIEX[idx[idx2[0]],1]
    else: 
        p2 = TAIEX[idx[idx3[0]],1] 
    profit.append(p1-p2)
print('************ strategy 2.1 ************')
show(profit)

#strategy3.0
profit = []
MN_profit = []
MN = []
mn = 0


for m in range(10,110,10):
    for n in range(10,110,10):
        if (m==n or m>n):
            MN.append([m,n])
            
    
def show_2(profit, tradeday):
    
    count =0
    max_profit = 0
    
    for j in range(len(MN)):
    
        for i in range(len(tradeday)):
            date = tradeday[i]
            idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
            idx.sort()
            p1 = TAIEX[idx[0],2]
            idx2 = np.nonzero(TAIEX[idx,4]<=p1-MN[j][0])[0] #停損
            idx3 = np.nonzero(TAIEX[idx,3]>=p1+MN[j][1])[0] #停利
            if(len(idx2)==0 and len(idx3)==0):
                p2 = TAIEX[idx[-1],1]
            elif(len(idx3)==0): #沒有停利點只有停損點
                p2 = TAIEX[idx[idx2[0]],1] 
            elif(len(idx2)==0): #沒有停損點只有停利點
                p2 = TAIEX[idx[idx3[0]],1]
            #都有的話就比早，看停損停利誰比較早發生
            elif(idx2[0]<idx3[0]):
                p2 = TAIEX[idx[idx2[0]],1]
            else: 
                p2 = TAIEX[idx[idx3[0]],1]
            
            
            profit.append(p2-p1)
            
        profit2 = np.cumsum(profit)
        count_profit = profit2[-1]
        
        if (count_profit>max_profit):
            max_profit = count_profit
            times = count
            mn = MN[j]
        
        count +=1
        MN_profit.append(profit)
        
    profit = MN_profit[times]        
    profit = np.array(profit)
    profit2 = np.cumsum(profit)
    ans0 = len(tradeday) #總進場次數
    ans1 = profit2[-1] #總損益點數
    ans2 = np.sum(profit>0)/len(profit) #勝率
    ans3 = np.mean(profit[profit>0])#賺錢時獲利點數
    ans4 = np.mean(profit[profit<=0])#輸錢時損失點數
    print('[m,n] = ',mn,'\nentry times:',ans0,'\ntotal:',ans1,'\nwin ratio:',ans2,'\nwin avg:',ans3,'\nlose avg:',ans4,'\nprofit distribution:')
    plt.hist(profit,bins=100)
    plt.show()
    print('accumulated profit:')
    plt.plot(profit2)
    plt.show()

print('************ strategy 3.0 ************')
show_2(profit, tradeday)

#strategy3.1
profit = []
MN_profit = []
MN = []
mn = 0


for m in range(10,110,10):
    for n in range(10,110,10):
        if (m==n or m>n):
            MN.append([m,n])
            
    
def show_3(profit, tradeday):
    
    count =0
    max_profit = 0
    
    for j in range(len(MN)):
    
        for i in range(len(tradeday)):
            date = tradeday[i]
            idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
            idx.sort()
            p1 = TAIEX[idx[0],2]
            idx2 = np.nonzero(TAIEX[idx,4]>=p1+MN[j][1])[0] #停損
            idx3 = np.nonzero(TAIEX[idx,3]<=p1-MN[j][0])[0] #停利
            if(len(idx2)==0 and len(idx3)==0):
                p2 = TAIEX[idx[-1],1]
            elif(len(idx3)==0): #沒有停利點只有停損點
                p2 = TAIEX[idx[idx2[0]],1] 
            elif(len(idx2)==0): #沒有停損點只有停利點
                p2 = TAIEX[idx[idx3[0]],1]
            #都有的話就比早，看停損停利誰比較早發生
            elif(idx2[0]<idx3[0]):
                p2 = TAIEX[idx[idx2[0]],1]
            else: 
                p2 = TAIEX[idx[idx3[0]],1]
            
            
            profit.append(p1-p2)
            
        profit2 = np.cumsum(profit)
        count_profit = profit2[-1]
        
        if (count_profit>max_profit):
            max_profit = count_profit
            times = count
            mn = MN[j]
        
        count +=1
        MN_profit.append(profit)
        
    profit = MN_profit[times]        
    profit = np.array(profit)
    profit2 = np.cumsum(profit)
    ans0 = len(tradeday) #總進場次數
    ans1 = profit2[-1] #總損益點數
    ans2 = np.sum(profit>0)/len(profit) #勝率
    ans3 = np.mean(profit[profit>0])#賺錢時獲利點數
    ans4 = np.mean(profit[profit<=0])#輸錢時損失點數
    print('[m,n] = ',mn,'\nentry times:',ans0,'\ntotal:',ans1,'\nwin ratio:',ans2,'\nwin avg:',ans3,'\nlose avg:',ans4,'\nprofit distribution:')
    plt.hist(profit,bins=100)
    plt.show()
    print('accumulated profit:')
    plt.plot(profit2)
    plt.show()

print('************ strategy 3.1 ************')
show_3(profit, tradeday)



        
        
        