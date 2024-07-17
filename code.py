import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data=....

def get_features_targets(data):
    features[:,0] = ...
    features[:,1] = ....
    features[:,2] = ....
    features[:,3] = ....
    features[:,4] = ....
    targets = data[' ']
    return (features, targets)

features, targets = get_features_targets(data)
X_train, X_test, y_train, y_test = train_test_split(features, targets,random_state=0)
knn = KNeighborsClassifier(n_neighbors = 4) #choose classifier
KNN_fit = knn.fit(X_train, y_train)#train classifier
accuracy = KNN_fit.score(X_test, y_test) #Estimate the accuracy of the classifier on future data
print ('KNN score: {}\n'.format(accuracy))
cv_scores = cross_val_score(knn, features, targets)
print('Cross-validation scores (3-fold):', cv_scores)
print('Mean cross-validation score (3-fold): {:.3f}'.format(np.mean(cv_scores)))
