#Training a model for the HR Dept. Using LR Algorithm with multiple variables, to predict salary for an emp based on some factors

import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n

df = pd.read_csv("ML/hiring.csv")
#print(df)


df.experience = df.experience.fillna("zero")
#print(df)

df.experience = df.experience.apply(w2n.word_to_num)
#print(df)

import math
median_test = math.floor(df['test_score(out of 10)'].mean())
print(median_test)