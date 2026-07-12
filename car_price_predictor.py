import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import SGDRegressor
from sklearn import metrics

data = pd.read_csv('car_dataset_2.csv')

data.head()

data.info()
data.isnull().sum()

p_data = data.dropna()
p_data.isnull().sum()

p_data.info()
p_data.fuel.value_counts()
p_data.seller_type.value_counts()
p_data.transmission.value_counts()
p_data.owner.value_counts()

p_data.replace({'fuel':{'Petrol':0,'Diesel':1,'CNG':2,'LPG':3,'Electric':4}},inplace=True)
p_data.replace({'seller_type':{'Individual':0,'Dealer':1,'Trustmark Dealer':2}},inplace=True)
p_data.replace({'transmission':{'Manual':0,'Automatic':1}},inplace=True)
p_data.replace({'owner':{'First Owner':0,'Second Owner':1,'Third Owner':2,'Fourth & Above Owner':3,'Test Drive Car':4}},inplace=True)

p_data.head()