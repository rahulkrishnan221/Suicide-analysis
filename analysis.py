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


import matplotlib.pyplot as plt
df.plot('Gender_binary','Number of suicides')
plt.show()
print(df.columns.values)
