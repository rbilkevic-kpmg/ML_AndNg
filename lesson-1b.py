import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

path = 'data\ex1data2.txt'
data = pd.read_csv(path, header=None, names=['area', 'rooms', 'price'])
tmp = data.loc[:, ['area', 'price']]
data.loc[:, ['area', 'price']] = (tmp - tmp.mean()) / tmp.std()

fig, ax = plt.subplots(figsize=(12, 8))
groups = data.groupby('rooms')
for name, group in groups:
    ax.scatter(group.area,
               group.price,
               edgecolor='black',
               s=100,
               alpha=0.7,
               label=name)
ax.legend(loc=2, title='Number of rooms')
ax.set_xlabel('House area')
ax.set_ylabel('House price')
ax.set_title('Area vs Price')
plt.show()


def compute_cost(theta, x, y):
    m = len(y)
    term = np.dot(x, theta) - y
    return 1 / (2 * m) * np.dot(term.T, term)


def gradient_descent(theta, x, y, a, n):
    m = len(y)
    cost = np.zeros(n)
    cost[0] = compute_cost(theta, x, y)
    for i in range(n):
        errors = np.dot(x, theta) - y
        theta = theta - a / m * np.dot(x.T, errors)
        cost[i] = compute_cost(theta, x, y)
    return theta, cost


# Set X and y
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]
X.insert(0, 'const', 1)

# Conversion to arrays
X = np.array(X)
y = np.array(y)

# Initialize theta and set hyper parameters
theta = np.zeros([X.shape[1], 1])
alpha = 0.003
niters = 2000

theta, cost_h = gradient_descent(theta, X, y, alpha, niters)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(niters), cost_h)
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


lin_reg = LinearRegression()
lin_reg.fit(X[:, 1:], y)

print('Gradient descent coefficients:')
print(theta.flatten())
print('Linear Regression coefficients')
print(lin_reg.intercept_.tolist() + lin_reg.coef_.flatten().tolist())