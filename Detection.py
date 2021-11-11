# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st 
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


st.title('Model Deployment: RandomForestClassifier')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Gender = st.sidebar.selectbox('Gender',('1','0'))
    ccs_procedure_code= st.sidebar.number_input("Procedure Code")
    ccs_diagnosis_code= st.sidebar.number_input("Diagnosis Code")
    Age = st.sidebar.number_input("Age")
    Days_spend_hsptl = st.sidebar.number_input("Days Spend in hospital")
    Charge = st.sidebar.number_input("Total Charge")
    data = {'Gender':Gender,
            'ccs_procedure_code' : ccs_procedure_code,
            'ccs_diagnosis_code':ccs_diagnosis_code,
            'Age':Age,
            'Days_spend_hsptl':Days_spend_hsptl,
            'Tot_charg':Charge}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)



df=pd.read_csv('Insurance Dataset.csv')
df

#from google.colab import drive
#drive.mount('/content/drive')

df.shape

df.size

#Data type
df.info()

df1=df.drop(['Hospital Id','Area_Service','Cultural_group','ethnicity','Abortion','Tot_cost','Weight_baby'],axis=1)

df1.shape

"""## Duplicates"""

#Count of duplicated rows
df1[df1.duplicated()].shape

#Print the duplicated rows
df1[df1.duplicated()]

#remove the duplicated rows
data=df1.drop_duplicates()

data.size

"""## Missing Values"""

data[data.isnull().any(axis=1)]

data = data.rename({'Mortality risk': 'mortalityrisk','ratio_of_total_costs_to_total_charges':'ratio_ofcost_to_charge','Home or self care,':'home_or_selfcare','Emergency dept_yes/No':'emergency_dept' }, axis=1)

data.isnull().sum()

data['mortalityrisk'].fillna(data.mortalityrisk.median(),inplace=True)

plt.figure(figsize=(20,20))
sns.heatmap(data.isnull(),yticklabels=False,cmap='viridis')

df=data.dropna()

df.isnull().sum()

#Datatype conversion
df['mortalityrisk']=df['mortalityrisk'].astype('int64')
#df['HospitalId']=df['HospitalId'].astype('int64')
#df['Tot_charg']=df['Tot_charg'].astype('int64')
#df['Tot_cost']=df['Tot_cost'].astype('int64')
#df['ratio_ofcost_to_charge']=df['ratio_ofcost_to_charge'].astype('int64')
#df

df.dtypes

df['Days_spend_hsptl'].replace('120 +', 120,inplace=True)

df['Days_spend_hsptl']=pd.to_numeric(df['Days_spend_hsptl'])

df['Days_spend_hsptl'].unique()

x=df.iloc[:, [0,1,2,4,5,8,11,12]]
x

from sklearn.preprocessing import LabelEncoder
Labelencoder_X= LabelEncoder()
X=x.apply(LabelEncoder().fit_transform)
X

X.dtypes

data=df.drop(["Hospital County","Age","Gender","Admission_type","home_or_selfcare","apr_drg_description","Surg_Description","emergency_dept"],axis=1)

fraud=pd.concat([X,data],axis=1)
fraud

bar = fraud.groupby("Result").count().iloc[:,0]

sns.barplot(x = bar.index, y=bar.values)

fraud.shape

"""## Feature Selection(Extra Trees Classifier)"""

X = fraud.drop(['Result'],axis = 1)# independent features
y = fraud['Result']# dependent feature
X.drop(['Surg_Description'],axis=1)

y.value_counts(normalize=True)

#model

from imblearn.over_sampling import SMOTE
smote=SMOTE()
x_train_os, y_train_os = smote.fit_resample(X,y)

from collections import Counter
print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_train_os)))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x_train_os,y_train_os,test_size=0.20,random_state=None)

class_weight=dict({0:100,1:1})
classifier=RandomForestClassifier(class_weight=class_weight)
classifier.fit(x_train_os,y_train_os)

prediction = classifier.predict(df)
prediction_proba = classifier.predict_proba(df)

st.subheader('Predicted Result')
st.write('Geniune' if prediction_proba[0][1] > 0.5 else 'Fraud')

st.subheader('Prediction Probability')
st.write(prediction_proba)