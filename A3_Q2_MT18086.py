#!/usr/bin/env python
# coding: utf-8

# ### Question 1)

# #### i) In the figure, goog200 is going upwards as per the day(unit time) is increasing so this will result variability in mean across the time series (not constant mean) thats why data is NOT stationary.

# #### iiI) The time series which is given have variablity in mean during two time intervals 1950-1965 and  1970-1980 therefore the data is not stationary.
# 

# #### v) Because the timeseries is getting decreased over time, so the mean is not constant because of variability in mean over the time make data non stationary. 

# #### ix)  Because the timeseries is getting increased over time, so the mean is not constant because of variability in mean over the time make data non stationary.
# 

# In[ ]:





# ### Question 2)

# #### Imports

# In[321]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import linregress

from statsmodels.tsa.stattools import adfuller

import statsmodels as sm
from statsmodels.tsa.arima_model import ARMA
from datetime import date
from datetime import datetime

import seaborn as sns 
sns.set()


# In[322]:


import csv
import os
import pandas as pd
import numpy as np
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.parser import parse 
from statsmodels.tsa.arima_model import ARMA
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# #### Part i)

# In[165]:


def makeplot(file):
    AMZN = pd.read_csv(file)
    fig = make_subplots(rows=3, cols=2, subplot_titles=("Stock Open Graph", "Stock High Graph", 
                                                        "Stock Low Graph", "Stock Close Graph",
                                                        "Stock Adj Close Graph", "Stock Volume Graph"))

    features = list(AMZN.columns)
    xlabel= features[0]
    features.remove(xlabel)
    i=1;j=1
    for ylabel in features:  
        fig.add_trace(go.Scatter(x=AMZN[xlabel], y=AMZN[ylabel]), row=i, col=j)
        fig.update_xaxes(title_text="Date( Year-Month )", row=i, col=j)
        fig.update_yaxes(title_text=ylabel, row=i, col=j)
        fig.update_layout(title_text="Customizing Subplot Axes", height=700)
        i+=1
        if i == 4:
            i = 1; j += 1
    if file == 'AMZN.csv':
        fig.update_layout(height=1500, width=1000, title_text="Time Series Graph Plots of Amazon Dataset")
    elif file == 'GOOG.csv':
        fig.update_layout(height=1500, width=1000, title_text="Time Series Graph Plots of Google Dataset")
    elif file == 'MSFT.csv':
        fig.update_layout(height=1500, width=1000, title_text="Time Series Graph Plots of Microsoft Dataset")
    fig.show()


# In[166]:


files = ['AMZN.csv', 'GOOG.csv', 'MSFT.csv'] 


# In[167]:


for i in files:
    makeplot(i)


# #### Part ii) 
# #####              a) Autocorrelation

# In[168]:


def makeAutocorrelationnplot(file):
    AMZN = pd.read_csv(file)
        
    fig, axes = plt.subplots(3,2, figsize=(15,20))
    
    features = list(AMZN.columns)
    xlabel= features[0]
    features.remove(xlabel)
    i=0;j=0
    for ylabel in features:  
        fig = plot_acf(AMZN[ylabel], lags= 100, ax=axes[i, j])
        s = "Stock " +ylabel+" Autocorrelation Graph for " +str(file.split('.')[0])+ " Dataset"
        axes[i,j].set_title(s)
        axes[i,j].set_xlabel("Lags")
        s1 = ylabel+" Autocorrelation" 
#         fig.show()
        axes[i,j].set_ylabel(s1)
        i+=1
        if i == 3:
            i = 0; j += 1
    fig.show()


# In[169]:


for i in files:
    makeAutocorrelationnplot(i)


# ##### Only volume feture of each dataset is stationary others are non stationary. 

# ##### b) Partial Autocorrelation

# In[170]:


def makepartialautocorelationplot(file):
    AMZN = pd.read_csv(file)
        
    fig, axes = plt.subplots(3,2, figsize=(15,20))
    
    features = list(AMZN.columns)
    xlabel= features[0]
    features.remove(xlabel)
    i=0;j=0
    for ylabel in features:  
        fig = plot_pacf(AMZN[ylabel], lags= 100, ax=axes[i, j])
        s = "Stock " +ylabel+" Autocorelation Graph" +str(file.split('.')[0])+ " Dataset"
        axes[i,j].set_title(s)
        axes[i,j].set_xlabel("Lags")
        s1 = ylabel+" Autocorelation" 
#         fig.show()
        axes[i,j].set_ylabel(s1)
        i+=1
        if i == 3:
            i = 0; j += 1
    fig.show()


# In[171]:


for i in files:
    makepartialautocorelationplot(i)


# #### iii) Stationarity Test

# In[172]:


def stationarityTest(file):
    AMZN = pd.read_csv(file)

    features = list(AMZN.columns)
    xlabel= features[0]
    features.remove(xlabel)
    i=0;j=0
    for feature in features: 
        X = AMZN[feature].values
        result = adfuller(X)
        print("Stationarity Test Result For ", file,"'s features ",feature)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        


# In[173]:


for i in files:
    stationarityTest(i)


# #### iv)

# #### Data Read

# In[323]:


train_data_file = ['train_data/AMZN_train.csv', 'train_data/GOOG_train.csv', 'train_data/MSFT_train.csv']


# In[324]:


test_data_file = ['test_data/AMZN_test.csv', 'test_data/GOOG_test.csv', 'test_data/MSFT_test.csv']


# In[ ]:





# In[325]:


#features extraction
df = pd.read_csv(test_data_file[0])
features = list(df.columns)
features.remove("Date")


# In[326]:


def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')

def train_test(train_loc,test_loc):
    train = pd.read_csv(train_loc, header=0, parse_dates=[0], index_col=0, squeeze=True,date_parser = parser)
    test = pd.read_csv(train_loc, header=0, parse_dates=[0], index_col=0, squeeze=True,date_parser = parser)
    return train,test


# In[327]:


def get_attribute_series(train,test,feature):
    feature = [feature]
    list_features = ["High","Low","Close","Adj Close","Open","Volume"]
    list_difference = list(set(list_features) - set(feature))
    for i in list_difference:
        train = train.drop(columns=i)
        test = test.drop(columns=i)
    return train,test


# In[328]:


# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


# In[382]:


def arma_train(train_series,test_series, feature):
    model = ARMA(train_series, (9,0))
    model_fit = model.fit(disp=False)
    
    print(model_fit.summary())
    residuals = DataFrame(model_fit.resid)
    residuals.plot(title = "Residual Error")
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())

    startdate = pd.datetime(2018,9,22)
    enddate = pd.datetime(2019,10,22)
    pred = model_fit.predict(startdate, enddate, dynamic = True)
    pred = pred.iloc[31:]   
    print(pred)
    
    return pred,test


# In[383]:


# print(np.asarray(train_series))


# In[384]:


train_loc = train_data_file[0]
test_loc = test_data_file[0]
train, test = train_test(train_loc,test_loc)

#interpolate both train and test
alldates = pd.date_range('22-10-2015', '19-10-2018')
train =train.reindex(alldates, fill_value = None)
train.head()
train = train.interpolate(method = 'time')

alldates = pd.date_range('22-10-2015', '19-10-2018')
test =test.reindex(alldates, fill_value = None)
test.head()
test = test.interpolate(method = 'time')

#Make data stationary other than Volume features...........
traindf = []
for feature in features:
    if feature != "Volume": 
        traindf.append((train[feature]-train[feature].shift(1)).dropna())
    else:
        traindf.append(train[feature])
        
testdf = []
for feature in features:
    if feature != "Volume": 
        testdf.append((test[feature]-test[feature].shift(1)).dropna())
    else:
        testdf.append(test[feature])


# In[385]:


test.head()


# ### Result of AMZN Dataset

# In[386]:


# for train, test, feature in zip(traindf, testdf, features):
#     arma_train(train, test, feature)
arma_train(train['Open'],test['Open'], features[0])


# In[295]:


test.head()


# In[297]:





# In[ ]:





# In[158]:


train_loc = train_data_file[1]
test_loc = test_data_file[1]
train, test = train_test(train_loc,test_loc)
#Handle missing dates and enter value in place of NULL
alldates = pd.date_range('22-10-2015', '19-10-2018')
train =train.reindex(alldates, fill_value = None)
train.head()
train = train.interpolate(method = 'time')
train.head()
train_series,test_series = get_attribute_series(train,test,"Volume")


# #### Result of GOOGLE Dataset

# In[202]:


model,result = arma_train(train_series,test_series)


# In[203]:


train_loc = train_data_file[2]
test_loc = test_data_file[2]
train, test = train_test(train_loc,test_loc)
#Handle missing dates and enter value in place of NULL
alldates = pd.date_range('22-10-2015', '19-10-2018')
train =train.reindex(alldates, fill_value = None)
train.head()
train = train.interpolate(method = 'time')
train.head()
train_series,test_series = get_attribute_series(train,test,"Volume")


# #### Result of Microsoft Dataset

# In[204]:


model,result = arma_train(train_series,test_series)


# #### Result = Because dataset is not stationary other than "Volume" feaure. So the results are not good for that fetures even after converting the data into stationary form will result inconsistent.   

# In[ ]:





# In[ ]:




