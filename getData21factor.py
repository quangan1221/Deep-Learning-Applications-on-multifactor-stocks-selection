import data_tool as dt
import sys
sys.path.append('../lib')
from db_info import *
import time
import os
import pandas as pd
import numpy as np

cfg = {
    "dataSource": {
        "mysql": {
            "mysql_sdfx": {
              
            },
            "mysql_fxcdb": {
            }
        }
    },
    "data": {
        "tradeDayList": {
            "src": "mysql",
            "conn": "mysql_fxcdb",
            "query": '''select TDATE
                        from TRADEDATE
                        where exchange = 'CNSESH'
                        and TDATE >= %(beginDate)s
                        and TDATE <= %(endDate)s
                        order by TDATE'''
        },
        "multiFactor": {
            "src": "mysql",
            "conn": "mysql_sdfx",
            "query": '''select *
                    from %(db_table)s
                    where TDATE >= str_to_date(%(beginDate)s,'%%Y%%m%%d')
                    and TDATE <= str_to_date(%(endDate)s,'%%Y%%m%%d')
                    order by TDATE'''
        }
    }
}
data_source = dt.DataApi(cfg)

factor1 = ['STK.ABS.BP','STK.ABS.DivYield','STK.ABS.DivYieldLY','STK.ABS.EP','STK.REL.RPB','STK.REL.RPE','STK.ABS.SalesToEV','STK.ABS.FCFFToEV']
factor2 =   ['STK.GRO.Ds2ev','STK.GRO.NetProfitGrowth','STK.GRO.RevenueGrowth'，'STK.MKT.LogTcap','STK.TECH.ILLIQ']

factor3 = ['STK.QUAL.Acca','STK.QUAL.Acca_OperFinanInvest','STK.QUAL.ROETTM','STK.QUAL.CashFromSalesToOperatingRevenueTTM']
factor4 = ['STK.TECH.Mon1','STK.TECH.Mon3','STK.TECH.Skewness_1Y_Daily','STK.TECH.VoturnoverChange_Mean_1M','STK.TECH.TurnOverAvg_1M','STK.TECH.TurnOverAvg_1M3M']


"""""""""""""""""""""           
'factor': {
    #'VALUE1':['STK.ABS.BP','STK.ABS.DivYield','STK.ABS.DivYieldLY','STK.ABS.EP','STK.REL.RPB','STK.REL.RPE','STK.ABS.SalesToEV','STK.ABS.FCFFToEV'],
    #'GRO':['STK.GRO.Ds2ev','STK.GRO.NetProfitGrowth','STK.GRO.RevenueGrowth'],
    #'MKT1':['STK.MKT.LogTcap','STK.TECH.ILLIQ'
    'QUAL1':['STK.QUAL.Acca','STK.QUAL.Acca_OperFinanInvest','STK.QUAL.ROETTM','STK.QUAL.CashFromSalesToOperatingRevenueTTM'],
    #'REVER1':['STK.TECH.Mon1','STK.TECH.Mon3','STK.TECH.Skewness_1Y_Daily'],
    #'TURN':['STK.TECH.VoturnoverChange_Mean_1M','STK.TECH.TurnOverAvg_1M','STK.TECH.TurnOverAvg_1M3M'],

    },
"""""""""""""""""""""

factor=[]
factor1.extend(factor4)         
print(factor1)  
         
os.chdir('C:/Users/Xuwen/Documents/sdfx_intern/deep learning/autoencoder')         
AdDay=pd.read_csv('adjustDay.csv',engine='python',skipfooter=3)           


daylist=np.array(AdDay['Start']).astype(str)
            
path = 'C:/Users/Xuwen/Documents/sdfx_intern/deep learning/autoencoder/data_demo/'


for line in factor3:
    #print(os.path.exists(path +line))
    if os.path.exists(path +line)==False:
        os.mkdir(path+line)
        #print(path+line)
    
    table = 'yj_gpyz_factordata_' + line.split('.')[-1].lower()
    #dateList = data_source.get(
        #'tradeDayList', beginDate='20080202', endDate='20080630')
    for tdate in daylist:
        if os.path.exists(path +line+ '/' + tdate + '.csv'):
            print(line +' '+ tdate + 'existed')
        else:
            data = data_source.get('multiFactor',db_table=table,beginDate=tdate,endDate=tdate)
            data.to_csv(path +line+ '/' + tdate + '.csv', index = False)
            print(line +' '+ tdate)
    


