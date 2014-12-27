'''Model using SVM.'''

import pandas as pd
import numpy as np
from sklearn import svm
import csv as csv

def handle_features(dataframe):
    # Making female and male integers
    dataframe['Gender'] = dataframe['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Handling missing values
    median_age = dataframe['Age'].dropna().median()
    if len(dataframe.Age[ dataframe.Age.isnull() ]) > 0:
        dataframe.loc[ (dataframe.Age.isnull()), 'Age'] = median_age

    if len(dataframe.Fare[ dataframe.Fare.isnull() ]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):                                              # loop 0 to 2
            median_fare[f] = dataframe[ dataframe.Pclass == f+1 ]['Fare'].dropna().median()
        for f in range(0,3):                                              # loop 0 to 2
            dataframe.loc[ (dataframe.Fare.isnull()) & (dataframe.Pclass == f+1 ), 'Fare'] = median_fare[f]

    # Removing some features not usefull for prediction
    dataframe = dataframe.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Embarked'], axis=1)
    print dataframe
    return dataframe

train_df = pd.read_csv('data/train.csv', header=0)
test_df = pd.read_csv('data/test.csv', header=0)
ids = test_df['PassengerId'].values

train_df = handle_features(train_df)
test_df = handle_features(test_df)

print test_df

train_data = train_df.values
test_data = test_df.values

print 'Training...'
train_X = train_data[0::,1::]
train_Y = train_data[0::,0]

clf = svm.SVC()
clf.fit(train_X, train_Y)

print 'Predicting...'
output = clf.predict(test_data).astype(int)


predictions_file = open("submissions/svm.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
