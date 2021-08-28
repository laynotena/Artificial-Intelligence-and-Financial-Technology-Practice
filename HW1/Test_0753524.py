# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
T = int(input())
for t in range(T):
    N = int(input())
    buy = []
    sell = []
    stockprice = 0
    for n in range(N):
        str = input()
        strlist = str.split()
        share = int(strlist[1])
        price = int(strlist[-1])
        if(strlist[0]=='buy'):
            while len(sell)>0:
                order = sell[0]
                if(order[0]>price):
                    break
                dealno = min(share,order[1])
                stockprice = order[0]
                order[1] -= dealno
                share -= dealno
                if(order[1]==0):
                    del sell[0]
                if(share==0):
                    break
            if(share>0):
                i = 0
                while((i<len(buy))and(price<buy[i][0])):
                    i+=1
                if((i<len(buy))and(price==buy[i][0])):
                    buy[i][1]+=share
                else:
                    buy.insert(i,[price,share])
        else:
            while len(buy)>0:
                order = buy[0]
                if(price> order[0]):
                    break
                dealno = min(share,order[1])
                stockprice = price
                order[1] -= dealno
                share -= dealno
                if(order[1]==0):
                    del buy[0]
                if(share==0):
                    break
            if (share>0):
                i=0
                while((i<len(sell))and(price>sell[i][0])):
                    i+=1
                if((i<len(sell))and(price==sell[i][0])):
                    sell[i][1]+=share
                else:
                    sell.insert(i,[price,share])
        
        if len(buy) > 0:
            bid_price = buy[0][0]
        else:
            bid_price = '-'

        if len(sell) > 0:
            ask_price = sell[0][0]
        else:
            ask_price = '-'

        if stockprice != 0:
            sold_price = stockprice
        else:
            sold_price = '-'

        print(ask_price, bid_price, sold_price)
