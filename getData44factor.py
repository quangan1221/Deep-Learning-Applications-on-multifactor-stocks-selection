import data_tool as dt
import sys
sys.path.append('../lib')
from db_info import *
import time
import os

cfg = {
    "dataSource": {
        "mysql": {
                       },
           
            }
        }
    },
    "data": {
        "tradeDayList": {
           
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

factor1 = ['STK.ABS.BP','STK.ABS.DivYield','STK.ABS.DivYieldLY','STK.ABS.EP','STK.ABS.PB','STK.ABS.PE','STK.ABS.SalesToEV',\
            'STK.ABS.FCFFToEV', 'STK.TECH.CloseAdjToOneMonthMaxCloseAdj','STK.TECH.Mon1','STK.TECH.Voturnover']
factor2 =   ['STK.TECH.VoturnoverChange_Mean_1M',\
            'STK.TECH.AmountAvg_1M','STK.TECH.Mon3','STK.TECH.Skewness_1Y_Daily','STK.TECH.TurnOverAvg_1M','STK.TECH.TurnOverAvg_1M3M',\
            'STK.TECH.ILLIQ','STK.MOM.Exmom11','STK.MOM.Mom5','STK.MOM.Mom8','STK.MOM.Mom11','STK.MKT.LogTcap','STK.MKT.Tcap',\
            'STK.MKT.MKT_FreeShares','STK.QUAL.Acca']

factor3 = ['STK.QUAL.Acca_OperFinanInvest','STK.QUAL.AssetImpairmentLossToGrossRevenue',\
            'STK.QUAL.CashFromSalesToOperatingRevenueTTM','STK.QUAL.CFOtoNOIttm','STK.QUAL.ROETTM','STK.QUAL.ROIC','STK.REL.RPB','STK.REL.RPE']
factor4 = ['STK.GRO.Ds2ev','STK.GRO.Fes2','STK.GRO.Lsg','STK.GRO.NetProfitGrowth','STK.GRO.OperProfitGrowth',\
            'STK.GRO.OperProfittoOperIncomeGrowth','STK.GRO.RevenueGrowth','STK.GRO.ROEGrowth','STK.GRO.Ssg','STK.GRO.ChangeOfNetProfGrowth']

path = "/media/fanyingjie/LENOVO/new/stk_multi_factor_research_v2/data/data_demo/"
for line in factor4:
    #print(os.path.exists(path +line))
    if os.path.exists(path +line)==False:
        os.mkdir(path+line)
        #print(path+line)
    
    table = 'yj_gpyz_factordata_' + line.split('.')[-1].lower()
    dateList = data_source.get(
        'tradeDayList', beginDate='20080101', endDate='20080131')
    for tdate in dateList.TDATE:
        if os.path.exists(path +line+ '/' + tdate + '.csv'):
            print(line +' '+ tdate + existed)
        else:
            data = data_source.get('multiFactor',db_table=table,beginDate=tdate,endDate=tdate)
            data.to_csv(path +line+ '/' + tdate + '.csv', index = False)
            print(line +' '+ tdate)
    


