#!/usr/bin/env python
# coding: utf-8

# # Twitter Sentimental Analysis

# Submitted By : Ansh Kumar Garg 
# 
# Section : ML
# 
# University Roll : 2015014    

# ### Importing the Libraries 

# In[1]:


import pandas as pd # Used for data analysis
import numpy as np # Used 
import re # Used for regular expressions 
import nltk # Used for text manipulation 
import warnings 
warnings.filterwarnings('ignore') # This will ignore all warnings  


# ### Loading the Dataset

# In[2]:


# importing a CSV file to DataFrame 
train = pd.read_csv(r"C:\Users\Ansh Garg\Desktop\Projects\Ansh_Twitter_Sentimental_Analysis_project\Data Set\train.csv")
test = pd.read_csv(r"C:\Users\Ansh Garg\Desktop\Projects\Ansh_Twitter_Sentimental_Analysis_project\Data Set\test.csv")


# ### Joining both the data sets

# In[3]:


# Joining test and train and storing in dataset and checking the head
dataset = train.append(test, ignore_index = True) #This will combine test and trai dataset
dataset.head(20)


# ### Preprocessing of Data  

# In[4]:


# Creating a function with name clean and doing preprocessing 
def clean(text):
    text = text.lower() #Converting all tweets to lower case
    text = re.sub(r'@[a-z0-9]+', '',text) #Removing all the twitter handles
    text = re.sub(r'#', '',text) #Removing '#' Symbols
    text = re.sub(r'RT[\s]+', '',text) #Removing RT 
    return text
dataset['tweet'] = dataset['tweet'].apply(clean)

# Removing the stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
dataset['tweet'] = dataset['tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
dataset.head(20)


# In[5]:


dataset = dataset.fillna(0)
dataset.head(10)


# ### Model Training 

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df = 0.90, min_df=2, max_features = 1000, stop_words = 'english')
bag_of_words = bow_vectorizer.fit_transform(dataset['tweet'])
bag_of_words.shape


# In[7]:


from sklearn.model_selection import train_test_split
xtrain, x_test, ytrain, y_test = train_test_split(bag_of_words, dataset['label'], random_state=1,test_size=0.25)


# In[8]:


# checking shape of dataset
print("shape of xtrain is :",xtrain.shape)
print("shape of ytrain is :",ytrain.shape)
print("shape of x_test is :",x_test.shape)
print("shape of y_test is :",y_test.shape)


# #### Using Naive_Bayes

# In[9]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
NB = MultinomialNB().fit(xtrain, ytrain)
pred = NB.predict(x_test)
print('Accuracy of NB  classifier on training set: {:.2f}' ,format(NB.score(xtrain, ytrain)))
print('Accuracy of NB classifier on test set: {:.2f}' ,format(NB.score(x_test, y_test)))
cm = confusion_matrix(y_test, pred)
cm


# #### Using Logistic Regression

# In[10]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
logreg = LogisticRegression().fit(xtrain, ytrain)
pred = logreg.predict(x_test)
print('Accuracy of Logistic classifier on training set {:.2f}' ,format(logreg.score(xtrain, ytrain)))
print('Accuracy of Logistic classifier on test set {:.2f}' ,format(logreg.score(x_test, y_test)))
cm = confusion_matrix(y_test, pred)
cm


# In[ ]:




