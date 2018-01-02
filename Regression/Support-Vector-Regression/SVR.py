# Support Vector Regression

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, [2]].values

#Splitting the dataset into the Training Set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)"""

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_y.fit_transform(Y)

#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

#Predicting a new result with Polynomial Regression
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


#Visualising the Polynomial Regression Results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth Or Bluff(Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression Results(For Better Visualisation and Smoother curves)
X_grid = np.arange( min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth Or Bluff(Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


