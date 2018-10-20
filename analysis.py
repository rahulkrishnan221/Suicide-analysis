import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#Reading data
df=pd.read_csv('data.csv')

#Total suicide happend
print("Total Suicides")
print(df['Number of suicides'].sum())

#preprocessing data with F & M with 0 and 1
df["Gender_binary"]=df["Gender"].replace(to_replace="M",value=0)
df["Gender_binary"]=df["Gender_binary"].replace(to_replace="F",value=1)

#Two more attributes added Female_suicide & Male_suicide
df.loc[(df['Gender'] == "F") & (df['Number of suicides'] >0),"Female_suicide"]=df['Number of suicides']
df.loc[(df['Gender'] == "M") & (df['Number of suicides'] >0),"Male_suicide"]=df['Number of suicides']


#Filling NaN value with 0
df["Female_suicide"]=df["Female_suicide"].fillna(0)
df["Male_suicide"]=df["Male_suicide"].fillna(0)


#Total Female Suicides
print("Total Female suicides")
print(df["Female_suicide"].sum())


#Total Male Suicides
print("Total Male suicides")
print(df["Male_suicide"].sum())

#Preprocessing Mapping social status with number dict comprehension
x={k:v for (k,v) in zip(df['Social_Status'].unique(),[m for m in range(6)])}

#New feature with mapping with number
df['Social_status_number']=df['Social_Status'].map(x)


import matplotlib.pyplot as plt
df.plot('Gender_binary','Number of suicides',kind="scatter")
#plt.show()
#print(df.columns.values)
#Removing 0 suicides row and forming new dataframe
df_value=df.where(df['Number of suicides']>0)

#Removing NaN
df_value=df_value.dropna()

#Main Causes of suicides
#print(df_value['Causes'].unique())

#Preprocessing Mapping age-group with number dict comprehension
age_dict={k:v for (k,v) in zip(df['Age_group'].unique(),[m for m in range(len(df['Age_group'].unique()))])}

#New feature with mapping with number
df['Age_group_number']=df['Age_group'].map(age_dict)

#Preprocessing Mapping Professional profile with number dict comprehension
prof_dict={k:v for (k,v) in zip(df['Professional_Profile'].unique(),[m for m in range(len(df['Professional_Profile'].unique()))])}

#New feature with mapping with number
df['Professional_Profile_number']=df['Professional_Profile'].map(prof_dict)


#Preprocessing Mapping Education_status with number dict comprehension
edu_dict={k:v for (k,v) in zip(df['Education_status'].unique(),[m for m in range(len(df['Education_status'].unique()))])}

#New feature with mapping with number
df['Education_status_number']=df['Education_status'].map(edu_dict)

#Preprocessing Mapping Causes with number dict comprehension
causes_dict={k:v for (k,v) in zip(df['Causes'].unique(),[m for m in range(len(df['Causes'].unique()))])}

#New feature with mapping with number
df['Causes_number']=df['Causes'].map(causes_dict)

#Preprocessing Mapping Means adopted with number dict comprehension
adopt_dict={k:v for (k,v) in zip(df['Means adopted'].unique(),[m for m in range(len(df['Means adopted'].unique()))])}

#New feature with mapping with number
df['means_adopted_number']=df['Means adopted'].map(adopt_dict)


#breaking in X and Y

X=df[['Year','Causes_number','means_adopted_number','Education_status_number','Professional_Profile_number','Gender_binary', 'Social_status_number', 'Age_group_number',]]

y=df['Number of suicides']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
print(df.isnull().values.any())
#knnreg = KNeighborsRegressor(n_neighbors = 5).fit(X_train, y_train)
#logicreg= LogisticRegression().fit(X_train,y_train)
random_forest = RandomForestClassifier(n_estimators=1000).fit(X_train, y_train)
print(random_forest.predict(X_test))
print(random_forest.score(X_test, y_test))
#print(logicreg.predict(X_test))
#print(logicreg.score(X_test, y_test))
#print(knnreg.predict(X_test))
#print(knnreg.score(X_test, y_test))
