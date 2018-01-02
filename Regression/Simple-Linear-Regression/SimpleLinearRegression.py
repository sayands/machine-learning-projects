#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Splitting the dataset into the Training Set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state = 0)

#Fitting Simple Linear Regression To The Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , Y_train)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Predicting Test Set Results
Y_pred = regressor.predict(X_test)

#Visualising The Training Set Results
plt.scatter(X_train,Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train) , color = 'blue')
plt.title("Salary vs Experience (Trainng set)")
plt.xlabel('Year Of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising The Test Results
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, Y_pred , color='blue')
plt.title("Salary vs Experience(Test Set)")
plt.xlabel("Year Of Experience")
plt.ylabel("Salary")
plt.show()

