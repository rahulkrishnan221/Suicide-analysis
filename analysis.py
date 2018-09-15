import pandas as pd

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
plt.show()
print(df.columns.values)
#Removing 0 suicides row and forming new dataframe
df_value=df.where(df['Number of suicides']>0)

#Removing NaN
df_value=df_value.dropna()

#Main Causes of suicides
print(df_value['Causes'].unique())
