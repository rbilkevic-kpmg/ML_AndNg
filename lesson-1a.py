# Machine Learning Online Class - Lesson 1a: Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = 'data\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['area', 'price'])

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(data.area,
           data.price,
           alpha=0.5,
           c='r',
           edgecolor='black')
ax.set_xlabel('Area')
ax.set_ylabel('Price')
ax.set_title('Area vs Price')
plt.show()


# ========== Cost and Gradient descent ==========
def compute_cost(theta, x, y):
    m = len(y)
    term = (np.dot(x, theta) - y)
    return 1 / (2 * m) * np.dot(term.T, term)


def gradient_descent(theta, x, y, alpha, niter):
    m = len(y)
    cost = np.zeros(niter)
    cost[0] = compute_cost(theta, x, y)
    for i in range(niter):
        error = (np.dot(x, theta) - y)
        theta = theta - alpha / m * np.dot(x.T, error)
        cost[i] = compute_cost(theta, x, y)

    return theta, cost


# Adding constant column
data.insert(0, 'const', 1)
# data1.insert(2, 'size_sqrt', np.power(data1.area, 2))

# Set X and y. Transform to numpy arrays
cols = data.shape[1]
X = np.array(data.iloc[:, 0:cols - 1])
y = np.array(data.iloc[:, cols - 1:cols])

# Initialize for gradient descent
theta = np.zeros([X.shape[1], 1])
alpha = 0.024
n = 1000
theta, cost_hist = gradient_descent(theta, X, y, alpha, n)

# Cost function plot over iterations
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(n-10), cost_hist[10:], 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


x = np.linspace(data.area.min(), data.area.max(), 100)
f = theta[0, 0] + theta[1, 0] * x

# Line fitting
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, c='b', label='Prediction')
ax.scatter(data.area,
           data.price,
           alpha=0.5,
           c='r',
           edgecolor='black', label='Training data')
ax.legend(loc=2)
ax.set_xlabel('Area')
ax.set_ylabel('Price')
ax.set_title('Area vs Price')
plt.show()


# ============= Visualizing J(theta_0, theta_1) =============

theta0_vals = np.linspace(-8, 0, 100)
theta1_vals = np.linspace(0.5, 2.5, 100)
X_m, Y_m = np.meshgrid(theta0_vals, theta1_vals)
Z = np.zeros([len(theta0_vals), len(theta1_vals)])
for i, t0 in enumerate(theta0_vals):
    for j, t1 in enumerate(theta1_vals):
        th = np.array([[t0], [t1]])
        Z[j, i] = compute_cost(th, X, y)

# Create a simple contour plot with labels using default colors.  The
fig, ax = plt.subplots(figsize=(12, 8))
CS = ax.contour(X_m, Y_m, Z, 24)
ax.clabel(CS, inline=1, fontsize=10)
ax.plot(theta[0], theta[1], c='r', marker='x')
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_title('Contour plot for values of theta')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_m, Y_m, Z)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel(r'Cost function $J$')
plt.show()

