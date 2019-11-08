#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import copy 
import numpy as np
from sklearn.svm import SVC
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.manifold import TSNE
import warnings 
warnings.filterwarnings('ignore')


# In[ ]:





# In[12]:


def tsne_plot(dataset_main,label_main,title,color_num):
    #TSNE Plot for glass dataset
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(dataset_main)

    df_subset = pd.DataFrame()
    df_subset['X'] = tsne_results[:,0]
    df_subset['y']=label_main
    df_subset['Y'] = tsne_results[:,1]
    plt.figure(figsize=(6,4))
    plt.title(title)
    sns.scatterplot(
        x="X", y="Y",
        hue="y",
        palette=sns.color_palette("hls", color_num),
        data=df_subset,
        legend="full",
        alpha=1.0
    )
    
def convert_feature(feature):
    data = []
    for  i in range(1,len(feature)):
        data.append(float(feature[i]))
    return data

def append_in_dataset(feature,dataset):
    for i in range(len(feature)):
        dataset[i].append(feature[i])
    return dataset

def get_main_feature_nom(dict,feature):
    data = []
    for i in feature:
        data.append(dict[i])
    return data

def convert_nominal(feature):
    nominal_feature = list(set(feature[1:]))
    size_nom = len(nominal_feature)
    dict_nom = {}
    for i in range(size_nom):
        dict_nom[nominal_feature[i]] = i
    return get_main_feature_nom(dict_nom,feature[1:])
        


# In[ ]:





# In[13]:


result = pd.read_csv('Telco_Data.csv',header=None)
count =-1
for i in range(len(result)):
    try:
        count =count +1
        a = result[i]
    except:
        break

labels = np.array(result[20])[1:]
length_dataset = len(result)-1


# In[14]:


dataset = []
for i in range(length_dataset):
    dataset.append([])
for i in range(1,count):
    feature = result[i]
    try:
        
        if isinstance(float(feature[1]), float):
            feature_con = convert_feature(feature)
            dataset = append_in_dataset(feature_con,dataset)
            continue
    except:
        feature_con = convert_nominal(feature)
        dataset = append_in_dataset(feature_con,dataset)
        pass


# In[15]:


dict_labels = {}
for i in labels:
    try:
        dict_labels[i] = dict_labels[i] + 1
    except:
        dict_labels[i] = 1
    
    


# In[16]:


print("Before Oversampling : ",dict_labels)


# In[17]:


# tsne_plot(dataset,labels,"TSNE after Aftersampling",2)


# In[18]:


sm = SMOTE(random_state=2)
data_oversample, labels_oversample = sm.fit_sample(dataset, labels)


# In[19]:


# tsne_plot(data_oversample,labels_oversample,"TSNE after Oversampling",2)


# In[20]:


dict_labels = {}
for i in labels_oversample:
    try:
        dict_labels[i] = dict_labels[i] + 1
    except:
        dict_labels[i] = 1
    
    


# In[ ]:





# In[21]:


print("After Oversampling : ",dict_labels)


# In[22]:


def bootstrapping(train_data,train_label,test_data,test_label,classifier,iterations,sample_size):
    
    iterations_list = []
    accuracy = []
    cohen_score = []
    count = [int(i) for i in range(len(train_data))]
    for i in range(iterations):
        counts = random.sample(count,sample_size)
        train = [train_data[j] for j in counts]
        labels = [train_label[j] for j in counts]
        clf = copy.deepcopy(classifier)
        clf.fit(np.array(train),np.array(labels))
        accuracy.append(clf.score(test_data,test_label))
        iterations_list.append(i)
        predict = clf.predict(test_data)
#         print(predict)
        cohen_score.append(cohen_kappa_score(predict,np.array(test_label)))
#     print(cohen_score)
    plt.plot(iterations_list,cohen_score,'go-', label='Bootstrapping')
    plt.plot(iterations_list,accuracy,':',  label='Accuracy')
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.title("Plot for Accuracy & Kappa")
    plt.legend()
    plt.show()
        


# In[23]:


labels_main = []
for i in labels_oversample:
    if i=="Yes":
        labels_main.append(1)
    elif i == "No":
        labels_main.append(0)
labels_oversample_1 = copy.deepcopy(labels_main)


# In[24]:


len(labels_oversample)


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(data_oversample, labels_oversample_1, test_size=0.25, random_state=42)


# In[26]:


classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
bootstrapping(X_train,  y_train, X_test, y_test, classifier, 100,5000)


# In[ ]:





# In[ ]:




