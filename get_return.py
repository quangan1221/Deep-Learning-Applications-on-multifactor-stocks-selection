"""

Created on Tue Oct 18 11:15:45 2016

@author: Quan Gan
"""



import data_tool as dt
import pandas as pd
import numpy as np
import datetime 




##########Configuration#######################
class mysql_CH_read(object):
	# 
	host = 'rdsshj1fvzlh92268305.mysql.rds.aliyuncs.com'
	user = 'erafxcdb'
	passwd = 'EraFxcdbSdfxTz'
	db = 'fxcdb'
    
class mysql_CH(object):
	# 
	host = 'rdsshj1fvzlh92268305.mysql.rds.aliyuncs.com'
	user = 'erafxcdb'
	passwd = 'EraFxcdbSdfxTz'
	db = 'fxcdb'

    
cfg = { "dataSource": {
            "mysql": {
                "mysql_CH_read": {
                    "host": mysql_CH_read.host,
                    "user": mysql_CH_read.user,
                    "passwd": mysql_CH_read.passwd,
                    "db": mysql_CH_read.db
                }
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
            "symbol": {
                "src": "mysql", # 数据源
                "conn": "mysql_CH_read", # 数据库链接 需在上方的dataSource中配置
                "query": "select symbol\
                            from TQ_SK_FININDIC\
                           where tradedate = %(tradedate)s\
                           order by symbol"
            }
        }
       }

cfg.keys()

def split_date(day):
    date=day.split('-')
    date=date[0]+date[1]+date[2]
    return date
def cal_date(day,delta):
    dat=day.split('-')
    #dat=dat[0]+dat[1]+dat[2]
    l_day=datetime.date(int(dat[0]),int(dat[1]),int(dat[2]))
    e_day=l_day+delta
    e_day=e_day.strftime("%Y-%m-%d")
    eday=split_date(e_day)
    
    return eday
    
def Get_Daylist(S_day,L_day,period,delta): #获取换仓日列表
    Sday=split_date(S_day)
    Eday=cal_date(L_day,delta)
    df = data_source.get("trade_day_list",beginday = Sday,endday = Eday)
    l=df['TDATE'].tolist()
    num=[k*period for k in range(int(len(l)/period)+1)]
    DayList=[l[n] for n in num]
    return DayList
	
def Get_PeriodReturn(sym,Day_list):
    get_price=dt.GetPrice()
    df=pd.DataFrame()
    m=0
    for symb in sym:
        
        df2=pd.DataFrame()
        for k in range(len(Day_list)):
            price=get_price.run(symbol=symb,start_day=Day_list[k],end_day=Day_list[k],asset='stk')
		#price=get_price.run(symbol=symb,start_day='20080107',end_day='20080109',asset='stk')
            price['tclose']=price['tclose'].astype(float)
		#price['return']=(price['tclose'][-1]/price['topen'][0])-1
		#price['return']=price['tclose'].pct_change(1)
            df2=df2.append(price)
        df2['return']=df2['tclose'].pct_change(1)
        l=df2['return'].tolist()
        del l[0]
        l2=l+[np.NaN]
        df2['Return']=np.array(l2)
        df=df.append(df2)
        #print(df2)
        m+=1
        print(m)
    df['tdate']=df.index
    Ret=df.loc[:,['tdate','symbol','Return']]
    Ret['tdate']=Ret.index
    Ret=Ret.dropna(how='any')
    Ret=Ret.reset_index(drop=True)
    return Ret
	
def Get_label(Ret):
    Retu=pd.DataFrame()
    for name,group in Ret.groupby('tdate'):
        group=group.sort(columns='Return')
        price1=group.head(int(len(group)*0.8))
        price1['label']=[0 for k in range(len(price1))]
        price2=group.tail(len(group)-int(len(group)*0.8))
        price2['label']=[1 for k in range(len(price2))]
    #price1=price1.append(price2)
        Retu=Retu.append(price1)  
        Retu=Retu.append(price2)
    Retu=Retu.sort(columns=['tdate','symbol'])
    return Retu

###############Get Symbol List##############################
data_source = dt.DataApi(cfg)
df1 = data_source.get('symbol',tradedate='20080107')

# 返回pandas DataFrame格式的数据
sym=df1["symbol"].tolist()
del sym[727]
#print(sym)


##################获得换仓日列表################################
S_day='2008-01-07'#第一个换仓日
L_day='2008-01-28'#最后一个换仓日
Period=5      #回测以周为换仓周期
Delta=datetime.timedelta(days=Period+2)#日期间隔算上周末
Day_list=Get_Daylist(S_day,L_day,Period,Delta)

#################获得从这个换仓日到下个换仓日期间的收益##############
Ret=Get_PeriodReturn(sym,Day_list)
Label=Get_label(Ret)
print(Label)
Retu.to_csv('Retu.csv')###测试期间存放在云端Retu.csv文件