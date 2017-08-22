# -*- coding: utf-8 -*-
"""

Created on Tue Oct 18 11:15:45 2016

@author: Quan Gan
"""
#逐个打开每个因子的文件夹
#######逐个打开每个时间段的数据csv，提取z-score，将后一天的数据衔接在前一天后。
####遍历完一个因子文件夹后，另起一列append另一个因子整个时间段的z-score

import os
import numpy as np
import pandas as pd
#import zipfile

#def idxfof(s,a):
#    idx=(a+'-'+s.split('\'')[1])
#    return idx
#
##提取因子/日期/股票的结构
##之后可以改为因子/股票/日期的结构
#def OneFactor(): #输出一个因子的全历史数据矩阵：股票*时间
#    path2='C:/Users/Xuwen/Documents/data_tool/data_demo/STK.ABS.BP'
#    os.chdir(path2)
#    files=[f for f in os.listdir('.') if os.path.isfile(f)]#每个因子有一个文件夹目录，逐个打开
#    Lof=len(files)
#    df1=pd.DataFrame()
##    if n<=1:
##        print(no data)
##    else:
##        date=str(df0['TDATE'][0])
##        date=date.split('-')
##        date=date[0]+date[1]+date[2]
##        zs=df0['ZSCORE']  #只要因子的ZSCORE一列
##        zs=np.array(zs).astype(float)
##        ss=ft.partial(idxfof,a='')
##        idex=list(df0['SYMBOL'].map(ss))
##        df1=pd.DataFrame(zs,index=idex,columns=[date])
#    for k in range(Lof):
#        if k%10==0:
#            print (k)
#        df=pd.read_csv(files[k])
#        n=len(df['TDATE'])
#        if n<=1:
#            continue  #忽略没有数据的空表，暂时的之后需要改成不忽略，设为NULL
#        else:
#            date=str(df['TDATE'][0])
#            date=date.split('-'); 
#            date=date[0]+date[1]+date[2]+'-'
#            zs=df['ZSCORE']  #只要因子的ZSCORE一列
#            zs=np.array(zs).astype(float)
#            ss=ft.partial(idxfof,a=date)
#            idex=list(df['SYMBOL'].map(ss))
#            df2=pd.DataFrame(zs,index=idex,columns=[date])
#            df1=pd.concat([df1,df2],axis=1)
#    #dff= df1.dropna(axis=0) #已经删去null
#    return df1

    
    
def OneFactor(factor_name): #输出一个因子的全历史数据单列：股票-时间
    #path2='C:/Users/Xuwen/Documents/data_tool/data_demo/STK.ABS.DivYield'
    #os.chdir(path2)
    files=[f for f in os.listdir('.') if os.path.isfile(f)]#每个因子有一个文件夹目录，逐个打开
    Lof=len(files)
    df1=pd.DataFrame()
    for k in range(Lof):
        print (k)
        df=pd.read_csv(files[k])
        n=len(df['TDATE'])
        if n<=1:
            continue  #忽略没有数据的空表，暂时的之后需要改成不忽略，设为NULL
        else:
            date=str(df['TDATE'][0])
            try:
                date=date.split('-')
                date=date[0]+date[1]+date[2];
            except IndexError:
                date=str(df['TDATE'][0])
                date=date.split('/')
                date=date[0]+date[1]+date[2]
            dat=np.array([date for k in range(len(df))])
            sym=[]
            for item in df['SYMBOL'].values:
                sym.append(item.split('\'')[1])
                

            zs=df['ZSCORE']  #只要因子的ZSCORE一列
            zs=np.array(zs).astype(float)
            #ss=ft.partial(idxfof,a=date)
            #idex=df['SYMBOL'].map(ss)
            df2=pd.DataFrame({'Date':dat,
                              'Symbol':np.array(sym),
                              'ZScore_'+factor_name: zs
                             }
                              )

            df1=df1.append(df2)
    #dff= df1.dropna(axis=0) #已经删去null
    return df1
#    
#def SToverTIME(df):
#    L=df.values.tolist()
#    flat=[]
#    for sub in L:
#       flat.extend(sub)
#    fl=pd.Series(flat)
#       
#    return fl
#    
#def TIMEoverST(df):
#    L=np.array(df)
#    LT=L.T
#    flat=[]
#    for sub in LT:
#       flat.extend(list(sub)) 
#    fl=pd.Series(flat)   
#    return fl    
#    

def GetData(path,method):
    os.chdir(path)
    pwd=os.getcwd()
    print('First Layer: ',pwd)
    name=os.listdir('.')
    folders=[d for d in name if os.path.isdir(d)]#遍历目录中的所有factor文件夹
    n=len(folders)
    
    #initialization
    pth=path+'/'+folders[0] #每次循环打开一个factor文件夹
    os.chdir(pth)
    pwd=os.getcwd()
    print('Second Layer:',pwd)
    Fac=OneFactor(folders[0])#存储单个因子的股票-时间许略
   
    for i in range(1,n):
        pth=path+'/'+folders[i] #每次循环打开一个factor文件夹
        os.chdir(pth)
        pwd=os.getcwd()
        print('Second Layer:',pwd)
        F=OneFactor(name[i+1])#存储单个因子的股票*时间矩阵
        if len(F)>0.75*len(Fac):    #######防止某因子数据过少而使得全体因子数据过少
            Fac=pd.merge(Fac,F,how='outer',on=['Date','Symbol'])#获得一个因子的历史数据
        else:
            continue
    os.chdir(path)
    pwd=os.getcwd()
    print('First Layer: ',pwd)    
    #Fac=Fac.dropna(axis=0,how='any')  #加入factor名字
    Result=Fac.dropna(how='any')
    if method=='SoverT':
        Result=Result.sort(columns=['Symbol','Date'])   #Stock over Time
        #Time over Stock 原顺序不变
        
    return Result          #写到csv里
    
if __name__ == "__main__":  
    
    path='C:/Users/Xuwen/Documents/sdfx_intern/deep learning/autoencoder/data_demo'
    Factor=GetData(path,method='ToverS')  #Stock Over Time
    
    
    Fac1=Factor.iloc[0:60000,:]
    Fac2=Factor.iloc[60000:120000,:]
    #Fac3=Factor.iloc[120000:120000,:]
    Fac3=Factor.iloc[120000:len(Factor),:]
    
#    Fac4=Factor.iloc[60000:80000,:]
#    Fac5=Factor.iloc[80000:100000,:]
#    Fac6=Factor.iloc[100000:120000,:]
#    Fac7=Factor.iloc[120000:140000,:]
#    Fac8=Factor.iloc[140000:len(Factor),:]
    Fac1.to_csv('Train1.csv',header=True,index=False)
    Fac2.to_csv('Train2.csv',header=True,index=False)
    Fac3.to_csv('Train3.csv',header=True,index=False)
    #Fac4.to_csv('train4.csv',header=True,index=False)
#    Fac5.to_csv('train5.csv',header=True,index=False)
#    Fac6.to_csv('train6.csv',header=True,index=False)
#    Fac7.to_csv('train7.csv',header=True,index=False)
#    Fac8.to_csv('train8.csv',header=True,index=False)
