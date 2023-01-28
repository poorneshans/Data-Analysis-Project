
image=img.imread('C:\\Users\\POORNESH N S\\Desktop\\W3\\euro.jpg')
plt.subplots(figsize=(20,7))
plt.subplot(111)
plt.text(25, 25, 'Euro vs US Dollar and Indian Rupee Exchange Rates: 1999-2022', fontsize=20,color='red',style='italic', bbox={
        'facecolor': 'White', 'alpha': 1, 'pad': 12})
plt.imshow(image)
plt.show()

import pandas as pd
df=pd.read_csv('E:\\EXTRA\\euro-daily-hist_1999_2022.csv')
df.head()
Period\Unit:	[Australian dollar ]	[Bulgarian lev ]	[Brazilian real ]	[Canadian dollar ]	[Swiss franc ]	[Chinese yuan renminbi ]	[Cypriot pound ]	[Czech koruna ]	[Danish krone ]	...	[Romanian leu ]	[Russian rouble ]	[Swedish krona ]	[Singapore dollar ]	[Slovenian tolar ]	[Slovak koruna ]	[Thai baht ]	[Turkish lira ]	[US dollar ]	[South African rand ]
0	2022-11-15	1.5415	1.9558	5.5480	1.3816	0.9790	7.3299	NaN	24.326	7.4388	...	4.9116	NaN	10.8081	1.4238	NaN	NaN	36.9390	19.3608	1.0404	17.8822
1	2022-11-14	1.5427	1.9558	5.4605	1.3706	0.9751	7.2906	NaN	24.289	7.4382	...	4.9043	NaN	10.7713	1.4177	NaN	NaN	36.9780	19.1923	1.0319	17.8393
2	2022-11-11	1.5459	1.9558	5.5147	1.3698	0.9844	7.3267	NaN	24.278	7.4384	...	4.8940	NaN	10.7241	1.4199	NaN	NaN	37.0880	19.0987	1.0308	17.7944
3	2022-11-10	1.5525	1.9558	5.2860	1.3467	0.9834	7.2184	NaN	24.361	7.4381	...	4.8913	NaN	10.8743	1.3963	NaN	NaN	36.7000	18.5100	0.9954	17.6882
4	2022-11-09	1.5538	1.9558	5.1947	1.3501	0.9880	7.2813	NaN	24.337	7.4382	...	4.9045	NaN	10.8450	1.4061	NaN	NaN	36.9990	18.6728	1.0039	17.8770
5 rows × 41 columns

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6177 entries, 0 to 6176
Data columns (total 41 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Period\Unit:              6177 non-null   object 
 1   [Australian dollar ]      6177 non-null   object 
 2   [Bulgarian lev ]          5775 non-null   object 
 3   [Brazilian real ]         5909 non-null   object 
 4   [Canadian dollar ]        6177 non-null   object 
 5   [Swiss franc ]            6177 non-null   object 
 6   [Chinese yuan renminbi ]  5909 non-null   object 
 7   [Cypriot pound ]          2346 non-null   object 
 8   [Czech koruna ]           6177 non-null   object 
 9   [Danish krone ]           6177 non-null   object 
 10  [Estonian kroon ]         3130 non-null   object 
 11  [UK pound sterling ]      6177 non-null   object 
 12  [Greek drachma ]          520 non-null    object 
 13  [Hong Kong dollar ]       6177 non-null   object 
 14  [Croatian kuna ]          5909 non-null   object 
 15  [Hungarian forint ]       6177 non-null   object 
 16  [Indonesian rupiah ]      6177 non-null   object 
 17  [Israeli shekel ]         5909 non-null   object 
 18  [Indian rupee ]           5909 non-null   object 
 19  [Iceland krona ]          3770 non-null   float64
 20  [Japanese yen ]           6177 non-null   object 
 21  [Korean won ]             6177 non-null   object 
 22  [Lithuanian litas ]       4159 non-null   object 
 23  [Latvian lats ]           3904 non-null   object 
 24  [Maltese lira ]           2346 non-null   object 
 25  [Mexican peso ]           6177 non-null   object 
 26  [Malaysian ringgit ]      6177 non-null   object 
 27  [Norwegian krone ]        6177 non-null   object 
 28  [New Zealand dollar ]     6177 non-null   object 
 29  [Philippine peso ]        6177 non-null   object 
 30  [Polish zloty ]           6177 non-null   object 
 31  [Romanian leu ]           6115 non-null   float64
 32  [Russian rouble ]         5994 non-null   object 
 33  [Swedish krona ]          6177 non-null   object 
 34  [Singapore dollar ]       6177 non-null   object 
 35  [Slovenian tolar ]        2085 non-null   object 
 36  [Slovak koruna ]          2608 non-null   object 
 37  [Thai baht ]              6177 non-null   object 
 38  [Turkish lira ]           6115 non-null   float64
 39  [US dollar ]              6177 non-null   object 
 40  [South African rand ]     6177 non-null   object 
dtypes: float64(3), object(38)
memory usage: 1.9+ MB
# data cleaning and formating:

df.rename(columns={'[US dollar ]':'us_dollar','[Indian rupee ]':'indian_rupee','Period\\Unit:':'date'},inplace=True)
col=df.columns.drop('date')

df[col]=df[col].apply(pd.to_numeric,errors='coerce')
df.date=pd.to_datetime(df.date)
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6177 entries, 0 to 6176
Data columns (total 41 columns):
 #   Column                    Non-Null Count  Dtype         
---  ------                    --------------  -----         
 0   date                      6177 non-null   datetime64[ns]
 1   [Australian dollar ]      6115 non-null   float64       
 2   [Bulgarian lev ]          5717 non-null   float64       
 3   [Brazilian real ]         5848 non-null   float64       
 4   [Canadian dollar ]        6115 non-null   float64       
 5   [Swiss franc ]            6115 non-null   float64       
 6   [Chinese yuan renminbi ]  5848 non-null   float64       
 7   [Cypriot pound ]          2304 non-null   float64       
 8   [Czech koruna ]           6115 non-null   float64       
 9   [Danish krone ]           6115 non-null   float64       
 10  [Estonian kroon ]         3074 non-null   float64       
 11  [UK pound sterling ]      6115 non-null   float64       
 12  [Greek drachma ]          514 non-null    float64       
 13  [Hong Kong dollar ]       6115 non-null   float64       
 14  [Croatian kuna ]          5848 non-null   float64       
 15  [Hungarian forint ]       6115 non-null   float64       
 16  [Indonesian rupiah ]      6115 non-null   float64       
 17  [Israeli shekel ]         5847 non-null   float64       
 18  indian_rupee              5848 non-null   float64       
 19  [Iceland krona ]          3770 non-null   float64       
 20  [Japanese yen ]           6115 non-null   float64       
 21  [Korean won ]             6115 non-null   float64       
 22  [Lithuanian litas ]       4097 non-null   float64       
 23  [Latvian lats ]           3842 non-null   float64       
 24  [Maltese lira ]           2304 non-null   float64       
 25  [Mexican peso ]           6115 non-null   float64       
 26  [Malaysian ringgit ]      6115 non-null   float64       
 27  [Norwegian krone ]        6115 non-null   float64       
 28  [New Zealand dollar ]     6115 non-null   float64       
 29  [Philippine peso ]        6115 non-null   float64       
 30  [Polish zloty ]           6115 non-null   float64       
 31  [Romanian leu ]           6115 non-null   float64       
 32  [Russian rouble ]         5932 non-null   float64       
 33  [Swedish krona ]          6115 non-null   float64       
 34  [Singapore dollar ]       6115 non-null   float64       
 35  [Slovenian tolar ]        2049 non-null   float64       
 36  [Slovak koruna ]          2560 non-null   float64       
 37  [Thai baht ]              6115 non-null   float64       
 38  [Turkish lira ]           6115 non-null   float64       
 39  us_dollar                 6115 non-null   float64       
 40  [South African rand ]     6115 non-null   float64       
dtypes: datetime64[ns](1), float64(40)
memory usage: 1.9 MB
# filling NAN with previous value

df.fillna(method='pad',inplace=True)
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6177 entries, 0 to 6176
Data columns (total 41 columns):
 #   Column                    Non-Null Count  Dtype         
---  ------                    --------------  -----         
 0   date                      6177 non-null   datetime64[ns]
 1   [Australian dollar ]      6177 non-null   float64       
 2   [Bulgarian lev ]          6177 non-null   float64       
 3   [Brazilian real ]         6177 non-null   float64       
 4   [Canadian dollar ]        6177 non-null   float64       
 5   [Swiss franc ]            6177 non-null   float64       
 6   [Chinese yuan renminbi ]  6177 non-null   float64       
 7   [Cypriot pound ]          2346 non-null   float64       
 8   [Czech koruna ]           6177 non-null   float64       
 9   [Danish krone ]           6177 non-null   float64       
 10  [Estonian kroon ]         3130 non-null   float64       
 11  [UK pound sterling ]      6177 non-null   float64       
 12  [Greek drachma ]          520 non-null    float64       
 13  [Hong Kong dollar ]       6177 non-null   float64       
 14  [Croatian kuna ]          6177 non-null   float64       
 15  [Hungarian forint ]       6177 non-null   float64       
 16  [Indonesian rupiah ]      6177 non-null   float64       
 17  [Israeli shekel ]         6177 non-null   float64       
 18  indian_rupee              6177 non-null   float64       
 19  [Iceland krona ]          6177 non-null   float64       
 20  [Japanese yen ]           6177 non-null   float64       
 21  [Korean won ]             6177 non-null   float64       
 22  [Lithuanian litas ]       4159 non-null   float64       
 23  [Latvian lats ]           3904 non-null   float64       
 24  [Maltese lira ]           2346 non-null   float64       
 25  [Mexican peso ]           6177 non-null   float64       
 26  [Malaysian ringgit ]      6177 non-null   float64       
 27  [Norwegian krone ]        6177 non-null   float64       
 28  [New Zealand dollar ]     6177 non-null   float64       
 29  [Philippine peso ]        6177 non-null   float64       
 30  [Polish zloty ]           6177 non-null   float64       
 31  [Romanian leu ]           6177 non-null   float64       
 32  [Russian rouble ]         5994 non-null   float64       
 33  [Swedish krona ]          6177 non-null   float64       
 34  [Singapore dollar ]       6177 non-null   float64       
 35  [Slovenian tolar ]        2085 non-null   float64       
 36  [Slovak koruna ]          2608 non-null   float64       
 37  [Thai baht ]              6177 non-null   float64       
 38  [Turkish lira ]           6177 non-null   float64       
 39  us_dollar                 6177 non-null   float64       
 40  [South African rand ]     6177 non-null   float64       
dtypes: datetime64[ns](1), float64(40)
memory usage: 1.9 MB
df.sort_values('date',inplace=True)

df.reset_index(inplace=True)

df.drop(columns='index',inplace=True)
df.head()
date	[Australian dollar ]	[Bulgarian lev ]	[Brazilian real ]	[Canadian dollar ]	[Swiss franc ]	[Chinese yuan renminbi ]	[Cypriot pound ]	[Czech koruna ]	[Danish krone ]	...	[Romanian leu ]	[Russian rouble ]	[Swedish krona ]	[Singapore dollar ]	[Slovenian tolar ]	[Slovak koruna ]	[Thai baht ]	[Turkish lira ]	us_dollar	[South African rand ]
0	1999-01-04	1.9100	1.9469	1.8718	1.8004	1.6168	8.5054	0.58231	35.107	7.4501	...	1.3111	25.2875	9.4696	1.9554	189.045	42.991	42.6799	0.3723	1.1789	6.9358
1	1999-01-05	1.8944	1.9469	1.8718	1.7965	1.6123	8.5054	0.58230	34.917	7.4495	...	1.3168	26.5876	9.4025	1.9655	188.775	42.848	42.5048	0.3728	1.1790	6.7975
2	1999-01-06	1.8820	1.9469	1.8718	1.7711	1.6116	8.5054	0.58200	34.850	7.4452	...	1.3168	27.4315	9.3050	1.9699	188.700	42.778	42.6949	0.3722	1.1743	6.7307
3	1999-01-07	1.8474	1.9469	1.8718	1.7602	1.6165	8.5054	0.58187	34.886	7.4431	...	1.3092	26.9876	9.1800	1.9436	188.800	42.765	42.1678	0.3701	1.1632	6.8283
4	1999-01-08	1.8406	1.9469	1.8718	1.7643	1.6138	8.5054	0.58187	34.938	7.4433	...	1.3143	27.2075	9.1650	1.9537	188.840	42.560	42.5590	0.3718	1.1659	6.7855
5 rows × 41 columns

# visualizations
import matplotlib.pyplot as plt
import numpy as np
# for euro-dollar exchange rate
plt.subplots(figsize=(15,5))
plt.subplot(1,2,1) 
plt.plot(df.date,df.us_dollar,'g')
plt.title('Daily euro-dollar exchange rate',fontsize=20,color='red')
plt.xlabel('Year',fontsize=15)
plt.ylabel('Dollar',fontsize=15)

# for euro-rupee exchange rate
plt.subplot(1,2,2)
plt.plot(df.date,df.indian_rupee,'b')
plt.title('Daily euro-rupee exchange rate',fontsize=20,color='red')
plt.xlabel('Year',fontsize=15)
plt.ylabel('Rupee',fontsize=15)
plt.show()

us_mean=df.us_dollar.rolling(365).mean()    # yearly mean value of dollar
in_mean=df.indian_rupee.rolling(365).mean() # yearly mean value of indian rupee
dyear=df.date.dt.year                       # extracted year from date
# each date,yearly mean value taken for euro-dollar exchange rate (1)
x=df.date
y=us_mean
plt.subplots(figsize=(15,5))
plt.subplot(2,2,1) 
plt.plot(x,y,'g')
plt.title('yearly mean value for each date:dollar',fontsize=20,color='red')
plt.xlabel('year',fontsize=15)
plt.ylabel('dollar',fontsize=15)

# each date, yearly mean value taken for euro-rupee exchange rate (2)
x=df.date
y=in_mean
plt.subplot(2,2,2)
plt.plot(x,y,'b')
plt.title('yearly mean value for each date:rupee',fontsize=20,color='red')
plt.xlabel('year',fontsize=15)
plt.ylabel('rupee',fontsize=15)

# year wise yearly mean value taken for euro-dollar exchange rate(3)
x=dyear
y=us_mean
plt.subplots(figsize=(15,5))
plt.subplot(2,2,3) 
plt.plot(x,y,'g')
plt.title('yearly wise mean euro-dollar exchange',fontsize=20,color='red')
plt.xlabel('year',fontsize=15)
plt.ylabel('dollar',fontsize=15)
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f},y={:.3f}".format(xmax,ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.94), **kw)
annot_max(x,y)
ax.set_ylim(0.8,1.8)

# year wise yearlyvmean value taken for euro-rupee exchange rate(4)
x=dyear
y=in_mean
plt.subplot(2,2,4)
plt.plot(x,y,'b')
plt.title('yearly wise mean euro-rupee exchange',fontsize=20,color='red')
plt.xlabel('year',fontsize=15)
plt.ylabel('rupee',fontsize=15)
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f},y={:.3f}".format(xmax,ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.34,0.94), **kw)
annot_max(x,y)
ax.set_ylim(20,100)
plt.show()


fig,ax=plt.subplots(figsize=(15,5))
ax.plot(df.date,us_mean,'g')
ax.set_xlabel('year',fontsize=12)
ax.set_ylabel('dollar',fontsize=12,color='green')

ax2=ax.twinx()
ax2.plot(df.date,in_mean,color='blue')
ax2.set_ylabel('rupee',fontsize=12,color='blue')
plt.title('Evolution of US dollar and Indian rupee vs Euro',fontsize=20,color='Red')
plt.grid()
plt.show()

import seaborn as sns
ddf=df[['date','us_dollar','indian_rupee']]
ddf['year']=df.date.dt.year
ddf['month']=df.date.dt.month
C:\Users\POORNESH N S\AppData\Local\Temp\ipykernel_23800\1393842896.py:2: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  ddf['year']=df.date.dt.year
C:\Users\POORNESH N S\AppData\Local\Temp\ipykernel_23800\1393842896.py:3: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  ddf['month']=df.date.dt.month
ddf=ddf.drop(columns='date')
ddf.head()
us_dollar	indian_rupee	year	month
0	1.1789	44.724	1999	1
1	1.1790	44.724	1999	1
2	1.1743	44.724	1999	1
3	1.1632	44.724	1999	1
4	1.1659	44.724	1999	1
plt.subplots(figsize=(15,8))
plt.subplot(1,2,1)
sns.heatmap(ddf.pivot_table(values='us_dollar',index='year',columns='month'))
plt.title('Euro vs Dollar',size=15)

plt.subplot(1,2,2)
sns.heatmap(ddf.pivot_table(values='indian_rupee',index='year',columns='month'))
plt.title('Euro vs Rupee',size=15)

print("\n> During feb 2008 to july 2008 euro-dollar exchange has peak rate")
print("> During sept 2013 & feb 2021 euro-rupee exchange has peak rate\n")
> During feb 2008 to july 2008 euro-dollar exchange has peak rate
> During sept 2013 & feb 2021 euro-rupee exchange has peak rate


plt.subplots(figsize=(25,5))
plt.title('Euro vs Dollar',size=20)
a=sns.barplot(x='year',y='us_dollar',data=ddf,palette='rainbow')
for i in a.containers:
    a.bar_label(i,)
    
print("\n> year 2008 has euro-dollar peak rate of 1.47 $")

plt.subplots(figsize=(25,5))
plt.title('Euro vs Rupee',size=20)
a=sns.barplot(x='year',y='indian_rupee',data=ddf,palette='rainbow')
for i in a.containers:
    a.bar_label(i,)

print("> year 2021 has euro-rupee peak rate of 87.43 ₹")
> year 2008 has euro-dollar peak rate of 1.47 $
> year 2021 has euro-rupee peak rate of 87.43 ₹


plt.subplots(figsize=(25,6))
sns.boxplot(x='year',y='us_dollar',data=ddf)
plt.title('Euro-dollar Exchange rate',color='black',size=20)

plt.subplots(figsize=(25,6))
sns.boxplot(x='year',y='indian_rupee',data=ddf)
plt.title('Euro-rupee Exchange rate',color='black',size=20)
Text(0.5, 1.0, 'Euro-rupee Exchange rate')


# us presidents terms of service
bill=ddf.loc[ddf.year.isin([1999,2000]) |((ddf.year==2001) & (ddf.month==1))] # only 2 years, not taken for graph
bush=ddf.loc[ddf.year.isin([2001,2002,2003,2004,2005,2006,2007,2008])|((ddf.year==2009) & (ddf.month==1))]
obama=ddf.loc[ddf.year.isin([2009,2010,2011,2012,2013,2014,2015,2016])|((ddf.year==2017)&(ddf.month==1))]
trump=ddf.loc[ddf.year.isin([2017,2018,2019,2020])|((ddf.year==2021)&(ddf.month==1))]
biden=ddf.loc[ddf.year.isin([2021,2022])]  # only 2 years, not taken for graph
# indian prime ministers terms of service
atal=ddf.loc[ddf.year.isin([1999,2000,2001,2002,2003])|((ddf.year==2004)&(ddf.month.isin([1,2,3,4,5])))]
singh=ddf.loc[ddf.year.isin([2005,2006,2007,2008,2009,2010,2011,2012,2013])|((ddf.year==2004)&(ddf.month.isin([6,7,8,9,10,11,12])))|((ddf.year==2014)&(ddf.month.isin([1,2,3,4,5])))]
modi=ddf.loc[ddf.year.isin([2015,2016,2017,2018,2019,2020,2021,2022])|((ddf.year==2014)&(ddf.month.isin([6,7,8,9,10,11,12])))]
plt.subplots(figsize=(20,4))
plt.subplot(231)
plt.title('George W Bush period',color='red',fontsize=20)
b=sns.lineplot(x='year',y='us_dollar',data=bush)
plt.grid()

plt.subplot(232)
plt.title('Barack Obama period',color='red',fontsize=20)
b=sns.lineplot(x='year',y='us_dollar',data=obama)
plt.grid()

plt.subplot(233)
plt.title('Donald Trump period',color='red',fontsize=20)
b=sns.lineplot(x='year',y='us_dollar',data=trump)
plt.grid()

plt.subplots(figsize=(50,4))
plt.subplot(234)
ax=sns.lineplot(x='year',y='us_dollar',data=ddf)
x=dyear
y=us_mean
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "year={:.3f},₹={:.3f}".format(xmax,ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=-90")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.74,0.93), **kw)
annot_max(x,y)
ax.set_ylim(0.8,1.6)
plt.title('Euro - Dollar Exchange rate Peaked at 1.4 during 2008 Financial Crises',color='green',size=20)
print('\n > Euro-Rupee Exchange rate during US-Presidents Period:')
 > Euro-Rupee Exchange rate during US-Presidents Period:


plt.subplots(figsize=(20,4))
plt.subplot(231)
plt.title('Atal Bihari Vajapeyi period',color='red',fontsize=20)
b=sns.lineplot(x='year',y='indian_rupee',data=atal)
plt.grid()

plt.subplot(232)
plt.title('Manmohan Singh period',color='red',fontsize=20)
b=sns.lineplot(x='year',y='indian_rupee',data=singh)
plt.grid()

plt.subplot(233)
plt.title('Narendra Modi period',color='red',fontsize=20)
b=sns.lineplot(x='year',y='indian_rupee',data=modi)
plt.grid()

plt.subplots(figsize=(50,4))
plt.subplot(234)
ax=sns.lineplot(x='year',y='indian_rupee',data=ddf)
x=dyear
y=in_mean
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "year={:.3f},₹={:.3f}".format(xmax,ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=-90")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.74,0.93), **kw)
annot_max(x,y)
ax.set_ylim(40,100)
plt.title('Euro - Rupee Exchange rate Peaked at 87.63 during 2021 rupee Depreciation',color='green',size=20)
print('\n > Euro-Rupee Exchange rate during indian Prime Ministers Period:')
 > Euro-Rupee Exchange rate during indian Prime Ministers Period:


import matplotlib.image as img 
print('\n> What Caused the US 2008 Financial Crisis?')
image=img.imread('C:\\Users\\POORNESH N S\\Desktop\\W3\\golden-scale-symbols-currencies-euro-260nw-126126335.webp')
plt.subplots(figsize=(30,5))
plt.subplot()
plt.imshow(image)
plt.show()
print('The 2008 financial crisis began with cheap credit and lax lending standards that fueled a housing bubble.')
print('When the bubble burst, the banks were left holding trillions of dollars of worthless investments in subprime mortgages.')
print('The Great Recession that followed cost many their jobs, their savings, and their homes.')

print('\n> What caused the india 2021 rupee Depreciation?')
image=img.imread('C:\\Users\\POORNESH N S\\Desktop\\W3\\golden-scale-symbols-currencies-euro-260nw-126126467.webp')
plt.subplots(figsize=(30,5))
plt.subplot()
plt.imshow(image)
plt.show()

print('Depreciation of rupee,caused by rising in import costs,threatening higher inflation and a widening trade deficit.')
print('The decline in the rupee in 2021 came after the US Federal Reserve and central banks of other advanced economies started')
print('aggressively raising interest rates to counter inflation.')
#print('The Fed increased interest rates by 425 basis points (bps) from near zero levels, between March and December, to 4.25-4.50%.')
print('This coupled with skyrocketing crude oil prices on account of geopolitical conflict in Europe led to a significant depreciation')
print('of rupee.The net effect of these opposing forces would determine the impact of a depreciating currency on an indian economy.')
> What Caused the US 2008 Financial Crisis?

The 2008 financial crisis began with cheap credit and lax lending standards that fueled a housing bubble.
When the bubble burst, the banks were left holding trillions of dollars of worthless investments in subprime mortgages.
The Great Recession that followed cost many their jobs, their savings, and their homes.

> What caused the india 2021 rupee Depreciation?

Depreciation of rupee,caused by rising in import costs,threatening higher inflation and a widening trade deficit.
The decline in the rupee in 2021 came after the US Federal Reserve and central banks of other advanced economies started
aggressively raising interest rates to counter inflation.
This coupled with skyrocketing crude oil prices on account of geopolitical conflict in Europe led to a significant depreciation
of rupee.The net effect of these opposing forces would determine the impact of a depreciating currency on an indian economy.
 

