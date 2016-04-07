# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pylab as P
from data_conversion import *

# Creating dataframe with CSV file
df = pd.read_csv('Data/data.csv', header=0);

# Normalising non-continuous parameters (given in string)
df['workclass'] = df['workclass'].map( lambda x: workclass(x) ).astype(int)
df['education'] = df['education'].map( lambda x: education(x) ).astype(int)
df['marital-status'] = df['marital-status'].map( lambda x: marital_status(x) ).astype(int)
df['occupation'] = df['occupation'].map( lambda x: occupation(x) ).astype(int)
df['relationship'] = df['relationship'].map( lambda x: relationship(x) ).astype(int)
df['race'] = df['race'].map( lambda x: race(x) ).astype(int)
df['sex'] = df['sex'].map( {'Female': 0, 'Male': 1} ).astype(int)
df['native-country'] = df['native-country'].map( lambda x: native_country(x) ).astype(int)
df['income'] = df['income'].map( lambda x: income(x) ).astype(int)

# Check the our dataframse is only containing numbers
df.dtypes[df.dtypes.map(lambda x: x=='object')]
#df['sex'] = df['sex'].map( lambda x: sex(x) ).astype(int)

# Delete unused columns
#df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
# Delete first column refering to user index
df = df.drop(df.columns[0], axis=1)

train_data = df.values
(row, col) = train_data.shape
for i in range(row):
    row = ""
    for j in range(col):
        row += str(train_data[i][j]) + " "
    print row

#
# # Get information
# print df.info()
# #print df.describe()
# # Get columns type
# #print df.dtypes
#
# # Filter columns
# #print df[(df['income'] == '>50K') & (df['sex'] == 'female')][['age']]
# print df[df['income'].isnull()][['age', 'sex', 'race']]
#
# df['age'].hist()
# P.show()
#
#
# # Display specific columns
# #print df[ ['age', 'workclass']]
#
# #
# # print type(df['age'])
# #
# # for age in df['age'][0:10]:
# #     print age
# # print df['age'][0:10]
