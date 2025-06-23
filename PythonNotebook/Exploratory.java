#data processing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
  
df=pd.read_csv("C:\Datasets\dataset.csv")
  
print(f"No of rows in the dataset --> {df.shape[0]}")
print(f"No of columns in the dataset --> {df.shape[1]}")
df.head(n=5)
  
print("the name of the columns in the dataset are : ")
df.columns

df.drop(["Time"],axis =1,inplace=True)
df.dtypes
df.isna().sum()

  df_fraud = df[df.Class==1]
df_true = df[df.Class==0]
df_true = df_true.sample(frac = 0.5)
data = pd.concat([df_true , df_fraud])
data = data.reset_index(drop = True)
data.shape
data['Amount'].describe()
with plt.style.context(('ggplot')):
    plt.figure(figsize = (8,5))
    plt.title("Distribution of classes")
    plt.plot(data["Amount"])
    plt.show()

  with plt.style.context(('ggplot')):
    plt.figure(figsize=(5,8))
    plt.title(" Distribution of fraud(1) and Non-fraud(0) transactions")
    sns.countplot(data = data , x = data["Class"])
    plt.show()

  fraud_per = round((len(df[df.Class==1])/len(df[df.Class==0]))*100,2)
print(f"the percentage of Fraudalent Transactions is {fraud_per}%")

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data.Amount=sc.fit_transform(data.Amount.values.reshape(-1,1))
data.drop_duplicates(inplace=True)
