# Data-Analysis-Project

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

df.info()

# data cleaning and formating:

df.rename(columns={'[US dollar ]':'us_dollar','[Indian rupee ]':'indian_rupee','Period\\Unit:':'date'},inplace=True)

col=df.columns.drop('date')

df[col]=df[col].apply(pd.to_numeric,errors='coerce')

df.date=pd.to_datetime(df.date)

df.info()

# filling NAN with previous value

df.fillna(method='pad',inplace=True)

df.info()

df.sort_values('date',inplace=True)

df.reset_index(inplace=True)

df.drop(columns='index',inplace=True)

df.head()

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

ddf=ddf.drop(columns='date')
ddf.head()

plt.subplots(figsize=(15,8))
plt.subplot(1,2,1)
sns.heatmap(ddf.pivot_table(values='us_dollar',index='year',columns='month'))
plt.title('Euro vs Dollar',size=15)

plt.subplot(1,2,2)
sns.heatmap(ddf.pivot_table(values='indian_rupee',index='year',columns='month'))
plt.title('Euro vs Rupee',size=15)

print("\n> During feb 2008 to july 2008 euro-dollar exchange has peak rate")
print("> During sept 2013 & feb 2021 euro-rupee exchange has peak rate\n")

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

plt.subplots(figsize=(25,6))
sns.boxplot(x='year',y='us_dollar',data=ddf)
plt.title('Euro-dollar Exchange rate',color='black',size=20)

plt.subplots(figsize=(25,6))
sns.boxplot(x='year',y='indian_rupee',data=ddf)
plt.title('Euro-rupee Exchange rate',color='black',size=20)

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

