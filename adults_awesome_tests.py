# -*- coding: utf-8 -*-
""" Predicting income for American workers
Authors: Jorge Ferreiro & Carlos Reyes.
"""
import pandas as pd
import numpy as np
import pylab as P
from data_conversion import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import csv as csv
from sklearn import svm

classifier_to_use = "super_vector"
classifier_to_use = "random_forest"

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

X = train_data[0:28000,0:13]
y = train_data[0:28000,14]

print '***********' * 10

if classifier_to_use == "random_forest":

    print 'Training...'

    forest = RandomForestClassifier(n_estimators = 200)
    forest = forest.fit(X, y)

    print 'Predicting...'
    predition = forest.predict(train_data[28001:30000,0:13]).astype(int)

    print "Score..."
    score = forest.score(train_data[28001:30000,0:13], train_data[28001:30000,14])
    score *= 100 # Porcentage
    print score

elif classifier_to_use == "super_vector":

    #clf = svm.SVC(kernel='rbf') #C=1.0, kernel='polynomial', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=-1, random_state=None)
    clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    print "CLF Done!"
    clf.fit(X, y)
    print "CLF FIT Done!"

    predition = clf.predict(train_data[28001:30000,0:13])
    print "CLF predictION Done!"


    x_svm = train_data[28001:30000,14]

    total = len(predition)
    ones = 0.0

    for i in range(total):
    	if predition[i] == x_svm[i]:
    		ones += 1.0;

    accuracy_svm = (ones/total)*100
    print "Accuracy SVM is " + str(accuracy_svm)

    # x = train_data[28001:30000,14]
    #
    # total = len(predition)
    # ones = 0.0
    #
    # for i in range(len(predition)):
    # 	if predition[i] == x[i]:
    # 		ones += 1.0;
    #
    # accuracy = (ones/total)*100
    #
    # print "Ones is " + str(ones)
    # print "Total is " + str(total)
    # print "Accuracy is " + str(accuracy)

print "****" * 30



predictions_file = open("adults_output.csv", "wb")
open_file_object = csv.writer(predictions_file)
# open_file_object.writerow(["AdultsId","Income"])
# open_file_object.writerows(zip(ids, output))
open_file_object.writerows(zip(predition))
predictions_file.close()
print 'Done.'
