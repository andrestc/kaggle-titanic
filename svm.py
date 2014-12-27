'''Model using SVM.'''

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
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


def fit_model_parameters(X, y):
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]}]

    scores = ['precision']
    best = None
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
        clf.fit(X_train, y_train)

        best = clf.best_estimator_

        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    return best


train_df = pd.read_csv('data/train.csv', header=0)
test_df = pd.read_csv('data/test.csv', header=0)
ids = test_df['PassengerId'].values

train_df = handle_features(train_df)
test_df = handle_features(test_df)

train_data = train_df.values
test_data = test_df.values

print 'Fitting model parameters..'
clf = fit_model_parameters(train_data[0::,1::], train_data[0::,0])

print 'Predicting...'
output = clf.predict(test_data).astype(int)

writeSubmissionFile("submissions/svm.csv", ids, output)
print 'Done.'
