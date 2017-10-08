# Machine Learning Online Class - Exercise 1: Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

path = 'data\ex1data1.txt'
data1 = pd.read_csv(path, header=None, names=['area', 'price'])

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(data1.area,
           data1.price,
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
data1.insert(0, 'const', 1)
# data1.insert(2, 'size_sqrt', np.power(data1.area, 2))

# Set X and y. Transform to numpy arrays
cols = data1.shape[1]
X = np.array(data1.iloc[:, 0:cols-1])
y = np.array(data1.iloc[:, cols-1:cols])

# Initialize for gradient descent
theta = np.zeros([X.shape[1], 1])
alpha = 0.022
n = 1000
theta, cost_hist = gradient_descent(theta, X, y, alpha, n)

# Cost function plot over iterations
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(n-10), cost_hist[10:], 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


x = np.linspace(data1.area.min(), data1.area.max(), 100)
f = theta[0, 0] + theta[1, 0] * x

# Line fitting
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, c='b')
ax.scatter(data1.area,
           data1.price,
           alpha=0.5,
           c='r',
           edgecolor='black')
ax.set_xlabel('Area')
ax.set_ylabel('Price')
ax.set_title('Area vs Price')
plt.show()


# %% ============= Part 4: Visualizing J(theta_0, theta_1) =============

# % Contour plot
# figure;
# % Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
# contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
# xlabel('\theta_0'); ylabel('\theta_1');
# hold on;
# plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

print(compute_cost(theta, X, y))

# Grid over which we will calculate J


theta0_vals = np.linspace(-8, 8, 100)
theta1_vals = np.linspace(-3, 5, 100)
X_m, Y_m = np.meshgrid(theta0_vals, theta1_vals)
Z = np.zeros([len(theta0_vals), len(theta1_vals)])
for i, t0 in enumerate(theta0_vals):
    for j, t1 in enumerate(theta1_vals):
        th = np.array([t0, t1]).reshape(2, 1)
        Z[i, j] = compute_cost(th, X, y)

# Create a simple contour plot with labels using default colors.  The
plt.contour(X_m, Y_m, Z)
plt.plot(*theta, c='r', marker='x')
plt.title('Contour plot for values of theta')
plt.show()
