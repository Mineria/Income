# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pylab as P
from data_conversion import *
from sklearn.ensemble import RandomForestClassifier
import csv as csv

# Creating dataframe with CSV file
data_df = pd.read_csv('Data/data.csv', header=0);
test_df = pd.read_csv('Data/test.csv', header=0) # Load the test file into a dataframe

def normalise_data(df, test_matrix):
    """Normalising non-continuous parameters (given in string)"""

    df['workclass'] = df['workclass'].map( lambda x: workclass(x) ).astype(int)
    df['education'] = df['education'].map( lambda x: education(x) ).astype(int)
    df['marital-status'] = df['marital-status'].map( lambda x: marital_status(x) ).astype(int)
    df['occupation'] = df['occupation'].map( lambda x: occupation(x) ).astype(int)
    df['relationship'] = df['relationship'].map( lambda x: relationship(x) ).astype(int)
    df['race'] = df['race'].map( lambda x: race(x) ).astype(int)
    df['sex'] = df['sex'].map( {'Female': 0, 'Male': 1} ).astype(int)
    df['native-country'] = df['native-country'].map( lambda x: native_country(x) ).astype(int)

    # For the test matrix there is no income
    # So we don't have to normalize it
    if not test_matrix:
        df['income'] = df['income'].map( lambda x: income(x) ).astype(int)

    # Check the our dataframse is only containing numbers
    df.dtypes[df.dtypes.map(lambda x: x=='object')]

    # Delete unused columns
    # Delete first column refering to user index
    df = df.drop(df.columns[0], axis=1)  
    #df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    return df.values

# Get Adults IDs from Test before delete the column
ids = test_df[test_df.columns[0]]

# Normalize Data and remove ID
train_data = normalise_data(data_df, test_matrix=False)
test_data  = normalise_data(test_df, test_matrix=True)

print test_data.shape

print 'Training...'
forest = RandomForestClassifier(n_estimators = 200)
forest = forest.fit(train_data[0:9999,0:13],train_data[0:9999,14])

# print 'Predicting...'
predition = forest.predict(train_data[10000:19999,0:13]).astype(int)
print type(train_data)

#print predition
#print "*"*14
#print train_data[11001:18000,14]
#print predition

print "*"*14


#score = forest.score(train_data[10000:19999,0:13], predition)
#print "Score..."
#print score

x = train_data[10000:19999,14]

total = len(predition)
ones = 0.0

for i in range(len(predition)):
	if predition[i] == x[i]:
		ones += 1.0;

accuracy = (ones/total)*100

print "Ones is " + str(ones)
print "Total is " + str(total)
print "Accuracy is " + str(accuracy)

# test_data_with_output = array(test_data)
# (test_rows, test_colums) = test_data.shape
# last_col = test_colums

# for i in range(test_rows):
#     test_data_with_output[i,last_col] 

predictions_file = open("adults_output.csv", "wb")
open_file_object = csv.writer(predictions_file)
# open_file_object.writerow(["AdultsId","Income"])
# open_file_object.writerows(zip(ids, output))
open_file_object.writerows(zip(predition))
predictions_file.close()
print 'Done.'


