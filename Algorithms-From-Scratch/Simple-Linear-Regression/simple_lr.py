# importing the libraries
import matplotlib.pyplot as plt 
import numpy as np 

# read the data and create matrices
my_data = np.genfromtxt('data.csv', delimiter=',') 
X = my_data[:, 0].reshape(-1,1)
ones = np.ones([X.shape[0], 1])
X = np.concatenate([ones, X], 1)
y = my_data[:, 1].reshape(-1, 1)

# plotting the scatter plot of the data
plt.scatter(my_data[:, 0].reshape(-1, 1), y)

# define the cost function
def computeCost(X, y, theta):
    inner = np.power(((X @ theta.T) -y), 2)
    return np.sum(inner) / ( 2 * len(X))



# gradient descent
def gradientDescent(X, y, theta, alpha, iters):
    for i in range(iters):
        theta = theta - (alpha / len(X)) * np.sum(( X @ theta.T - y) * X, axis = 0)
        cost = computeCost(X, y, theta)
        
        #if i % 10 == 0:
        #    print(cost)
    return (theta, cost)

# setting the hyperparameters
alpha = 0.0001
iters = 2000

# initialising the parameters
theta = np.array([[1.0, 1.0]])

# print(computeCost(X, y, theta))

g, cost = gradientDescent(X, y, theta, alpha, iters)
print (g, cost)

# plotting the equation along with the scatter plot of the data
plt.scatter(my_data[:, 0].reshape(-1, 1), y)
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = g[0][0] + g[0][1] * x_vals
plt.plot(x_vals, y_vals, '--')
