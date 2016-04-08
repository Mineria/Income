# -*- coding: utf-8 -*-
""" Predicting income for American workers
Authors: Jorge Ferreiro & Carlos Reyes.
"""
import pandas as pd
import numpy as np
from income import normalize
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import csv as csv
from sklearn import svm

# Filesname
output_file = 'output/adults_prediction.csv'

# Configuration options
n_jobs = -1
verbose = False
warm_start = False
n_estimators = 1000

calculate_score = True
classifier_to_use = "random_forest" # neighbors | super_vector | random_forest

# Creating dataframe with CSV file
data_df = pd.read_csv('data/data.csv', header=0) # Load the test file into a dataframe
test_df = pd.read_csv('data/test.csv', header=0)

def print_configuration():
    print
    print "*** Configuration " + "*" * 30
    print "Classifier: " + str(classifier_to_use or "No Classifier")
    print "Verbose: " + str(verbose or "No verbose")
    print "n_estimators: " + str(n_estimators or "No n_estimators")
    print "warm_start: " + str(warm_start or "No warm_start")
    print "n_jobs: " + str(n_jobs or "No n_jobs")
    print

def normalise_data(df, test_matrix):
    """Normalising non-continuous parameters (given in string)"""

    df['workclass'] = df['workclass'].map( lambda x: normalize.workclass(x) ).astype(int)
    df['education'] = df['education'].map( lambda x: normalize.education(x) ).astype(int)
    df['marital-status'] = df['marital-status'].map( lambda x: normalize.marital_status(x) ).astype(int)
    df['occupation'] = df['occupation'].map( lambda x: normalize.occupation(x) ).astype(int)
    df['relationship'] = df['relationship'].map( lambda x: normalize.relationship(x) ).astype(int)
    df['race'] = df['race'].map( lambda x: normalize.race(x) ).astype(int)
    df['sex'] = df['sex'].map( {'Female': 0, 'Male': 1} ).astype(int)
    df['native-country'] = df['native-country'].map( lambda x: normalize.native_country(x) ).astype(int)

    # For the test matrix there is no income
    # So we don't have to normalize it
    if not test_matrix:
        df['income'] = df['income'].map( lambda x: normalize.income(x) ).astype(int)

    # Check the our dataframse is only containing numbers
    df.dtypes[df.dtypes.map(lambda x: x=='object')]

    # Delete unused columns
    # Delete first column refering to user index
    df = df.drop(df.columns[0], axis=1)
    #df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    return df.values

print_configuration()

# Get Adults IDs from Test before delete the column
ids = test_df[test_df.columns[0]]

# Normalize Data and remove ID
train_data = normalise_data(data_df, test_matrix=False)
test_data  = normalise_data(test_df, test_matrix=True)

X = train_data[0:28000, 0:-2] #-2 = all features except result column
y = train_data[0:28000,-1] #-1 = last column with results (14)

X_predict = train_data[28001:30000, 0:-2] #-2 = all features except result column
y_predict = train_data[28001:30000, -1] #-1 = last column with results (14)

# X_predict = test_data[0::, 0:13]
# y_predict = []

if classifier_to_use == "random_forest":

    forest = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, verbose=verbose, warm_start=warm_start)#criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, random_state=None, verbose=1, warm_start=False, class_weight=None)

    print 'Training...'
    forest = forest.fit(X, y)

    print 'Predicting...'
    predition = forest.predict(X_predict).astype(int)

    if calculate_score:
        print "Score..."
        score = forest.score(X_predict, y_predict)
        score *= 100 # Porcentage
        print score

elif classifier_to_use == "super_vector":

    #clf = svm.SVC(kernel='rbf') #C=1.0, kernel='polynomial', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=-1, random_state=None)
    clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    print "CLF Done!"
    clf.fit(X, y)
    print "CLF FIT Done!"

    predition = clf.predict(X_predict)
    print "CLF predictION Done!"

    total = len(predition)
    ones = 0.0

    for i in range(total):
    	if predition[i] == y_predict[i]:
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

elif classifier_to_use == "neighbors":

    clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None)
    print "KNeighbors Done!"

    clf.fit(X, y)
    print "CLF FIT Done!"

    print 'Predicting...'
    predition = clf.predict(X_predict).astype(int)
    print "CLF predictION Done!"

    if calculate_score:
        print "Score..."
        score = clf.score(X_predict, y_predict)
        score *= 100 # Porcentage
        print score

print "****" * 20



predictions_file = open( output_file , "wb")
open_file_object = csv.writer(predictions_file)
# open_file_object.writerow(["AdultsId","Income"])
# open_file_object.writerows(zip(ids, output))
open_file_object.writerows(zip(predition))
predictions_file.close()
print 'Done.'
