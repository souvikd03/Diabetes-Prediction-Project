#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Importing Dataset
dataset = pd.read_csv("diabetes.csv")
dataset


# In[3]:


dataset.info


# In[4]:


dataset.isnull().sum()


# In[5]:


dataset.describe()


# In[6]:


#correlation of independent variables
plt.figure(figsize=(10,8))
sns.heatmap(dataset.corr(),annot=True,fmt=".3f",cmap="YlGnBu")
plt.title("correlation heatmap")


# In[8]:


#Exploring Pregnancy and target variables
plt.figure(figsize=(10,8))
#Plotting Density function graph of the pregnancies and target variables
kde = sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==1],color = "Red", fill = True)
kde = sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==0],color = "Blue", fill = True)
kde.set_xlabel("Pregnancies")
kde.set_ylabel("Density")
kde.legend(["Positive","Negative"])


# In[9]:


#Exploring Glucose and target variables
plt.figure(figsize=(10,8))
sns.violinplot(data=dataset,x="Outcome",y="Glucose",split=True, linewidth=2,inner = "quart")


# In[10]:


#Exploring Glucose and target variables
plt.figure(figsize=(10,8))
#Plotting Density function graph of the glucose and target variables
kde = sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==1],color = "Red", fill = True)
kde = sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==0],color = "Blue", fill = True)
kde.set_xlabel("glucose")
kde.set_ylabel("Density")
kde.legend(["Positive","Negative"])


# In[11]:


#Replace 0 values with the mean/median of the respective feature
#Glucose
dataset["Glucose"]=dataset["Glucose"].replace(0,dataset["Glucose"].median())
#Bloodpressure
dataset["BloodPressure"]=dataset["BloodPressure"].replace(0,dataset["BloodPressure"].median())
#BMI
dataset["BMI"]=dataset["BMI"].replace(0,dataset["BMI"].mean())
#SkinThickness
dataset["SkinThickness"]=dataset["SkinThickness"].replace(0,dataset["SkinThickness"].mean())
#Insulin
dataset["Isulin"]=dataset["Isulin"].replace(0,dataset["Isulin"].mean())


# In[12]:


#Split the features and label
x=dataset.drop(["Outcome"],axis=1)
y=dataset["Outcome"]


# In[13]:


x


# In[14]:


y


# In[15]:


#split train and test dataset
#!pip install scikit-learn
from sklearn.model_selection import train_test_split


# In[25]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)


# In[26]:


x_train


# In[18]:


#KNN
from sklearn.neighbors import KNeighborsClassifier


# In[33]:


training_accuracy=[]
test_accuracy = []
for n_neighbors in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train,y_train)
    
    #check accuracy score
    training_accuracy.append(knn.score(x_train,y_train))
    test_accuracy.append(knn.score(x_test,y_test))


# In[20]:


plt.plot(range(1,15),training_accuracy,label="training_accuracy")
plt.plot(range(1,15),test_accuracy,label = "test_accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# In[31]:


knn = KNeighborsClassifier(n_neighbors=14)
knn.fit(x_train,y_train)
print(knn.score(x_train,y_train),": Training accuracy")
print(knn.score(x_test,y_test),": Test accuracy")


# In[32]:


from sklearn.tree import DesicionTreeClassifier
dt=DesicionTreeClassifier(random_state=0,max_depth=7)
dt.fit(x_train,y_train)
print(dt.score(x_train,y_train),": Training accuracy")
print(dt.score(x_test,y_test),": Test accuracy")


# In[24]:


from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(random_state=42)
mlp.fit(X_train,y_train)
print(mlp.score(X_train,y_train),": Training accuracy")
print(mlp.score(X_test,y_test),": Test accuracy")


# In[22]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)


# In[23]:


mlp1=MLPClassifier(random_state=0)
mlp1.fit(X_train_scaled,y_train)
print(mlp1.score(X_train_scaled,y_train),": Training accuracy")
print(mlp1.score(X_test_scaled,y_test),": Test accuracy")
# Diabetes-Prediction-Project
