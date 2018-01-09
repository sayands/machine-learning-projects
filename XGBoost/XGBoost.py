# XG Boost Classifier

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('advertising.csv')
X = dataset[["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Male"]]
Y = dataset.iloc[: ,9].values

#Splitting the dataset into the Training Set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state = 0)


#Creating the XGBoost Classifier
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train,Y_train)

#Predicting Test Set Results
y_pred = model.predict(X_test)

#Making The Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

#Evaluating the model
from sklearn.metrics import classification_report
result = classification_report(Y_test,y_pred)

# Applying k-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 10)
accuracies.mean()
accuracies.std()
