#!/usr/bin/env python
# coding: utf-8

# ## Question 1

# ### Imports

# In[177]:


from lifelines import KaplanMeierFitter
from lifelines.datasets import load_waltons
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


# In[178]:


df = load_waltons()
df.head()


# In[179]:


controldata = df[df['group']=='control'] 
groupdata =  df[df['group']!='control']


# In[180]:


groupdata.head()


# In[181]:


#Seperate dataset
controldata = controldata.sort_values(by='T',na_position='first')
groupdata = groupdata.sort_values(by='T',na_position='first')
#set index to 0 to len(controldata)
controldata.index = np.arange(0,len(controldata)) 
groupdata.index = np.arange(0,len(groupdata)) 


# ### KM PLOT

# In[182]:


def make_kmp(controldata,length):
    newdf = pd.DataFrame(columns=['time','Nt', 'Dt', 'Ct','St'])
    previndex= 0
    currindex = 0
    tcolindex = 0
    day = 0
    nt = len(controldata)
    x = []
    for i in range(0,length):
        try: 
            tcolindex  += 1
            if length == currindex:
                break
            if i== 0:
                newdf.loc[day] = [day] + [nt] + [0] +  [0] + [1]
            elif controldata.loc[currindex]['T'] > day:  
                newdf.loc[day] = newdf.loc[day-1]
                newdf.loc[day]['time'] = day
                newdf.loc[day]['Dt'] = 0
                newdf.loc[day]['Ct'] = 0
                nt = nt - (newdf.loc[day-1]['Dt'] + newdf.loc[day-1]['Ct'])
                newdf.loc[day]['Nt'] = nt
            elif controldata.loc[currindex]['T'] == day:
                dt = 0; ct = 0
                val = controldata.loc[currindex]['T'] 
                while(val == controldata.loc[currindex]['T']):
                    if controldata.loc[currindex]['E'] == 1:
                        dt += 1
                    else:
                        ct += 1
                    currindex += 1
                    if len(controldata) == currindex:
                        break
                st = (newdf.loc[day-1]['St'])*float((nt-dt)/nt)
                nt = nt - (newdf.loc[day-1]['Dt'] + newdf.loc[day-1]['Ct'])
                newdf.loc[day] = [day] + [nt] + [dt] + [ct] + [st]

            day += 1
        except:
            newdf.loc[day] = newdf.loc[day-1]
            newdf.loc[day]['time'] = day
            newdf.loc[day]['Dt'] = 0
            newdf.loc[day]['Ct'] = 0
            nt = nt - (newdf.loc[day-1]['Dt'] + newdf.loc[day-1]['Ct'])
            newdf.loc[day]['Nt'] = nt
            
            day += 1
    return newdf


# In[183]:


def makeplot(newdf):
    plt.plot(list(newdf['time']), list(newdf['St']))
    plt.xlabel('Time') 
    plt.ylabel('Probability') 
    plt.title('KM survival plot') 
    plt.show()


# In[184]:


length = int(max(max(controldata['T']), max(groupdata['T'])) + 1)
newdf1 = make_kmp(controldata,length)
makeplot(newdf1)


# In[ ]:





# In[185]:


newdf2 = make_kmp(groupdata,length)
newdf2.fillna(0, inplace = True)
makeplot(newdf2)


# In[186]:


plt.plot(list(newdf1['time']), list(newdf1['St']), label='Control Group')
plt.plot(list(newdf2['time']), list(newdf2['St']), label='miR-137 Group')
plt.xlabel('Time') 
plt.ylabel('Probability') 
plt.title('KM survival plot') 
plt.legend(loc='upper right')
plt.show()


# ### LOG RANK

# In[ ]:





# In[234]:


def detlogrank(newdf1, newdf2):
    controldf = newdf1.filter(['time','Nt','Dt'], axis=1)
    controldf = controldf.rename(columns = {'Nt':"Nt1", "Dt":"Dt1"})
    datagroupdf = newdf2.filter(['Nt','Dt'], axis=1)
    datagroupdf = datagroupdf.rename(columns = {'Nt':"Nt2", "Dt":"Dt2"})
    data = pd.concat([controldf,datagroupdf], axis=1, sort=False)
    nt = data['Nt1'] + data['Nt2']
    dt = data['Dt1'] + data['Dt2']
    et1 = data['Nt1']*(dt/nt)
    et2 = data['Nt2']*(dt/nt)
    sum_dt1 = sum(list(data['Dt1'])[1:])
    sum_dt2 = sum(list(data['Dt2'])[1:])
    sum_et1 = sum(list(et1)[1:]) 
    sum_et2 = sum(list(et2)[1:])

    log_rank_test = float(((sum_dt1 - sum_et1)**2)/sum_et1) + float(((sum_dt2 - sum_et2)**2)/sum_et2)
    return log_rank_test


# In[235]:


log_rank_test = detlogrank(newdf1, newdf2)
print("Log Rank : ",log_rank_test)


# #### for the value of alpha which is 0.05 and df = 1 value from chi square table we get is 3.84. Our value of log rank is greater than the value 3.84 thus we can reject null hypothesis and can say that both curves for two different group are different with 95% confidence. 
# 

# ### Median Survival For Each Group

# In[190]:


def mediansurv(newdf1, newdf2):
    median_survival1 = list(newdf1['St'])
    median_survival2 = list(newdf2['St'])
    index1 = min(range(len(median_survival1)), key=lambda i: abs(median_survival1[i]-0.5))
    print("median survival for Control group : ",index1," days")
    index2 = min(range(len(median_survival2)), key=lambda i: abs(median_survival2[i]-0.5))
    print("median survival for miR-137 group : ",index2," days")


# In[191]:


mediansurv(newdf1, newdf2)


# In[116]:


# # plot using library
# T = df['T']
# E = df['E']

# T1 = controldata['T']
# E1 = controldata['E']
# kmf = KaplanMeierFitter()
# kmf.fit(T, event_observed=E)
# kmf.plot()
# T2 = groupdata['T']
# E2 = groupdata['E']
# kmf = KaplanMeierFitter()
# kmf.fit(T, event_observed=E)
# kmf.plot()
# from lifelines.statistics import logrank_test
# results = logrank_test(T1, T2, E1, E2, alpha=.05)
# results.print_summary()


# ## Question 2

# In[239]:


def makeexppoints(scalevalue):
    points = list(np.random.exponential(scale=scalevalue, size=100))
    exppoints = []
    for i in points:
        x = int(i) + 1
        exppoints.append(x)
    expdf = pd.DataFrame(columns=['T','E'])
    count = 0
    for i in exppoints:
        rand = random.uniform(0, 1)
        if rand > 0.1:
            e = 1
        else:
            e = 0
        expdf.loc[count] = [i] + [e] 
        count += 1
    expdf = expdf.sort_values(by='T',na_position='first')
    expdf.index = np.arange(0,len(expdf)) 
    return expdf


# In[240]:


expdf1 = makeexppoints(scalevalue=10)
expdf2 = makeexppoints(scalevalue=11)
length1 = max(max(list(expdf1['T'])),max(list(expdf2['T'])))


# In[241]:


length1


# In[242]:


newdf4 = make_kmp(expdf1,length1)
newdf4.fillna(0, inplace = True)
makeplot(newdf4)


# In[243]:


length1


# In[244]:


newdf5 = make_kmp(expdf2,length1)
newdf5.fillna(0, inplace = True)
makeplot(newdf5)


# In[ ]:





# In[245]:


plt.plot(list(newdf4['time']), list(newdf4['St']), label='Group 1')
plt.plot(list(newdf5['time']), list(newdf5['St']), label='Group 2')
plt.xlabel('Time') 
plt.ylabel('Probability') 
plt.title('KM survival plot') 
plt.legend(loc='upper right')
plt.show()


# In[247]:


log_rank_test = detlogrank(newdf4, newdf5)
print("Log Rank : ",log_rank_test)


# #### for the value of alpha which is 0.4674 and df = 1 value from chi square table we get is 3.84. Our value of log rank is less than the value 3.84 thus we can accept the null hypothesis and can say that both curves for two different group are same.

# In[248]:


mediansurv(newdf4, newdf5)


# In[ ]:




