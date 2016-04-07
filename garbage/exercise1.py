# -*- coding: utf-8 -*-
""" We have to predict the income for a given input test
Author : Jorge Ferreiro & Carlos Reyes.
Date : 4 April 2016
please see packages.python.org/milk/randomforests.html for more
"""
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('csv/data.csv', header=0)        # Load the train file into a dataframe

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# <=50K = 0, >50K = 1
train_df['income'] = train_df['income'].map( {'<=50K': 0, '>50K': 1} ).astype(int)

# All the ages with no data -> make the median of all Ages
median_age = test_df['age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age


print train_df
#
# Workclass = list(enumerate(np.unique(train_df['workclass'])))    # determine all values of Embarked,
# Workclass_dict = { name : i for i, name in Workclass }              # set up a dictionary in the form  Ports : index
# train_df.Embarked = train_df.Embarked.map( lambda x: Workclass_dict[x]).astype(int)     # Convert all Embark strings to int
