# Random Forest Regression

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#Splitting the dataset into the Training Set and Test set
'''from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)'''

#Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''

#Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,Y)

#Predicting a new result with Polynomial Regression
y_pred = regressor.predict(6.5)

#Visualising the Polynomial Regression Results(For Better Visualisation and Smoother curves)
X_grid = np.arange( min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth Or Bluff(Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()