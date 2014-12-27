'''Model using SVM.'''

import pandas as pd
import numpy as np
from sklearn import svm
import csv as csv

def handle_features(dataframe):
    # Making female and male integers
    dataframe['Gender'] = dataframe['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Making Embarked as Int
    Ports = list(enumerate(np.unique(dataframe['Embarked'])))
    Ports_dict = { name : i for i, name in Ports }
    dataframe.Embarked = dataframe.Embarked.map( lambda x: Ports_dict[x]).astype(int)

    # Handling missing values
    median_age = dataframe['Age'].dropna().median()
    if len(dataframe.Age[ dataframe.Age.isnull() ]) > 0:
        dataframe.loc[ (dataframe.Age.isnull()), 'Age'] = median_age

    if len(dataframe.Fare[ dataframe.Fare.isnull() ]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):
            median_fare[f] = dataframe[ dataframe.Pclass == f+1 ]['Fare'].dropna().median()
        for f in range(0,3):
            dataframe.loc[ (dataframe.Fare.isnull()) & (dataframe.Pclass == f+1 ), 'Fare'] = median_fare[f]

    # Removing some features not usefull for prediction
    dataframe = dataframe.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    return dataframe

def writeSubmissionFile(file_name, ids, output):
    print 'Printing submission file.'
    predictions_file = open(file_name, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()

train_df = pd.read_csv('data/train.csv', header=0)
test_df = pd.read_csv('data/test.csv', header=0)
ids = test_df['PassengerId'].values

train_df = handle_features(train_df)
test_df = handle_features(test_df)

train_data = train_df.values
test_data = test_df.values

print 'Training...'
clf = svm.SVC()
clf.fit(train_data[0::,1::], train_data[0::,0])

print 'Predicting...'
output = clf.predict(test_data).astype(int)

writeSubmissionFile("submissions/svm.csv", ids, output)
print 'Done.'
