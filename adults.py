import csv as csv
import pandas as pd

train_matrix = pd.read_csv('Data/data.csv', header=0)        # Load the train file into a dataframe

print train_matrix
