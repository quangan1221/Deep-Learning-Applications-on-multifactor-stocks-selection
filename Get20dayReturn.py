# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:29:52 2016

@author: Quan
"""
import os
path = 'C:/Users/Xuwen/Documents/sdfx_intern/deep learning/autoencoder'
os.chdir(path)

import data_tool as dt
import pandas as pd
import numpy as np
import datetime 




##########Configuration#######################

 
 
cfg = { "dataSource": {
            "mysql": {
              
            }
        },
        "data": {
            # mysql数据库中的数据
            "trade_day_list": { # 数据名称
                "src": "mysql", # 数据源
                "conn": "mysql_CH_read", # 数据库链接 需在上方的dataSource中配置
                # 查询语句
                "query": "select TDATE\
                                from TRADEDATE\
                               where exchange = 'CNSESH'\
                                 and TDATE >= %(beginday)s\
                                 and TDATE <= %(endday)s\
                               order by TDATE"
            },
            # 获取本地csv文件的配置方法
            "test_csv_data": {
                "src": "localFile",
                "mode": "csv_to_df",
                "url": "E:/Tool/dataApi"
            },
            "symbol": { # 数据名称
                "src": "mysql",
                "conn": "mysql_CH_read",
                "query": "select symbol\
                            from TQ_SK_FININDIC\
                           where tradedate = %(tradedate)s\
                           order by symbol"
            },
        }
    }
 

    
def Get_PeriodReturn(sym,Day_list):
    get_price=dt.GetPrice()
    df=pd.DataFrame()
    m=0
    for symb in sym:
        df2=pd.DataFrame()
        m+=1
        print(m)
        get_price=dt.GetPrice()
        try:
            for k in range(len(Day_list)):
                price=get_price.run(symbol=symb,start_day=Day_list[k],end_day=Day_list[k],asset='stk')
                price['tclose']=price['tclose'].astype(float)
                df2=df2.append(price)           
            #except mysql.connector.Error:
                #
        except Exception as e:
            print('error'+str(m))
            #traceback.print_exc() 
            continue
        df2['return']=df2['tclose'].pct_change(1)
        df2['return']=df2['return'].replace(np.inf,np.nan)
        l=df2['return'].tolist()
        del l[0]
        l2=l+[np.NaN]
        df2['Return']=np.array(l2)
        df=df.append(df2)
        #print(df2)
    df['tdate']=df.index
    Ret=df.loc[:,['tdate','symbol','Return']]
    Ret['tdate']=Ret.index
    Ret=Ret.dropna(how='any')
    Ret=Ret.reset_index(drop=True)
    return Ret   
    
    
    
def Get_3label(Ret):  #将股票分为五档 0%-20%:0, 20%-40%:1,40%-60%:2,60%-80%:3,80%-100%:4最强
    Retu=pd.DataFrame()
    for name,group in Ret.groupby('tdate'):
        #price1=group['Return'].quantile(.9)
        #print(price1)
        #print(len(group))
        price1=group[group['Return']>=group['Return'].quantile(0.67)]
        price1['label']=[0 for k in range(len(price1))]
        price2=group[(group['Return']>=group['Return'].quantile(0.33))&(group['Return']<group['Return'].quantile(0.67))]
        price2['label']=[1 for k in range(len(price2))]
        price3=group[group['Return']<group['Return'].quantile(0.33)]
        price3['label']=[2 for k in range(len(price3))]
    #price1=price1.append(price2)
        Retu=Retu.append(price1)  
        Retu=Retu.append(price2)
        Retu=Retu.append(price3)
    Retu=Retu.sort(columns=['tdate','symbol'])
    return Retu
    
def Get_5label(Ret):  #将股票分为五档 0%-20%:0, 20%-40%:1,40%-60%:2,60%-80%:3,80%-100%:4最强
    Retu=pd.DataFrame()
    for name,group in Ret.groupby('tdate'):
        #price1=group['Return'].quantile(.9)
        #print(price1)
        #print(len(group))
        price1=group[group['Return']>=group['Return'].quantile(0.8)]
        price1['label']=[0 for k in range(len(price1))]
        price2=group[(group['Return']>=group['Return'].quantile(0.6))&(group['Return']<group['Return'].quantile(0.8))]
        price2['label']=[1 for k in range(len(price2))]
        price3=group[(group['Return']>=group['Return'].quantile(0.4))&(group['Return']<group['Return'].quantile(0.6))]
        price3['label']=[2 for k in range(len(price3))]
        price4=group[(group['Return']>=group['Return'].quantile(0.2))&(group['Return']<group['Return'].quantile(0.4))]
        price4['label']=[3 for k in range(len(price4))]
        price5=group[group['Return']<group['Return'].quantile(0.2)]
        price5['label']=[4 for k in range(len(price5))]
    #price1=price1.append(price2)
        Retu=Retu.append(price1)  
        Retu=Retu.append(price2)
        Retu=Retu.append(price3)
        Retu=Retu.append(price4)
        Retu=Retu.append(price5)
    Retu=Retu.sort(columns=['tdate','symbol'])
    return Retu

def Get_2label(Ret,perc):  #将股票分为两档 0%-30%:0 弱, 70%-100%: 1 强
    Retu=pd.DataFrame()
    for name,group in Ret.groupby('tdate'):
        #price1=group['Return'].quantile(.9)
        #print(price1)
        #print(len(group))
        price1=group[group['Return']>=group['Return'].quantile(1-perc)]
        price1['label']=[1 for k in range(len(price1))]
        price5=group[group['Return']<=group['Return'].quantile(perc)]
        price5['label']=[0 for k in range(len(price5))]
    #price1=price1.append(price2)
        Retu=Retu.append(price1)  
        Retu=Retu.append(price5)
    Retu=Retu.sort(columns=['tdate','symbol'])
    return Retu

os.chdir('C:/Users/Xuwen/Documents/sdfx_intern/deep learning/data/')      
TdDay=pd.read_csv('tradeday_list.csv',engine='python') 

AdDay=TdDay[::10]
    
AdDay.to_csv('adjustDay10.csv')           

DayList=np.array(AdDay['TDATE']).astype(str)   
#######################Get 20 day return########################33
path2 = 'C:/Users/Xuwen/Documents/sdfx_intern/deep learning/data/20-day-return/'
os.chdir(path2)

Retu=pd.DataFrame()
os.chdir(path2)
for Day in DayList:
    print(Day)
    Ret=pd.read_csv(Day+'.csv',engine='python')
    Retu=Retu.append(Ret)
    
Retu=Retu.rename(columns={'date': 'tdate','20-day-return':'Return'})

#######################现成直接取######################################
 

#######################现场下载计算########################################3   
data_source = dt.DataApi(cfg)
######由未复权价格计算所得
path2 = 'C:/Users/Xuwen/Documents/sdfx_intern/deep learning/autoencoder/20-day-return/'    
os.chdir(path2)
df=pd.read_csv('20110114.csv')

# 返回pandas DataFrame格式的数据
sym=df["symbol"].tolist()
#del sym[727]
#print(sym)
##################获得换仓日列表################################
#S_day='2008-02-01'#第一个换仓日
#L_day='2008-06-30'#最后一天
path1 = 'C:/Users/Xuwen/Documents/sdfx_intern/deep learning/autoencoder/'    
os.chdir(path1)     
AdDay=pd.read_csv('adjustDay10.csv',engine='python',skipfooter=3)           
DayList=np.array(AdDay['TDATE']).astype(str)
Ret=Get_PeriodReturn(sym,DayList)

#Period=20      
#回测以20天为换仓周期
#Delta=datetime.timedelta(days=Period)#日期间隔算上周末
#print(Delta)
#Day_list=Get_Daylist(S_day,L_day,Period,Delta)
#print(Day_list)

#Ret=Get_PeriodReturn(sym[1000:1050],DayList)
#percent=0.3

#计算标签，label分为两档，换仓期间收益率表现为前20%的为强势股标为1，后20%为弱势标为label分为两档，换仓期间收益率表现为前20%的为强势股标为1，后20%为弱势标为0,数据平衡
Label2=Get_2label(Retu,.5)   
print(Label)
