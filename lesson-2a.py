# Machine Learning Online Class - Lesson 2a: Logistic Classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

path = 'data/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'accepted'])
labels = {0: 'Not Accepted', 1: 'Accepted'}

fig, ax = plt.subplots(figsize=(12, 8))
groups = data.groupby('accepted')
for name, group in groups:
    ax.scatter(group.exam1,
               group.exam2,
               s=50,
               alpha=0.7,
               label=labels[name])
ax.legend(loc=1)
ax.set_xlabel('Exam 1')
ax.set_ylabel('Exam 2')
ax.set_title('Scores of exams')
plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_cost(theta, x, y):
    m = x.shape[0]
    h_x = sigmoid(np.dot(x, theta))
    term_1 = np.dot(y.T, np.log(h_x))
    term_2 = np.dot((1 - y).T, np.log(1 - h_x))
    return -1/m * (term_1 + term_2)


def gradient_function(theta, x, y):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    m = x.shape[0]

    errors = sigmoid(np.dot(x, theta.T)) - y
    grad = np.dot(errors.T, x) / m

    return grad


def predict(x, theta):
    pred = sigmoid(np.dot(theta, x))
    pred = pred >= 0.5

    return pred


def plot_decision_boundary(data):
    x1 = np.linspace(data.exam1.min() - 20, data.exam1.max() + 20, 2)
    x2 = (-1 / theta[2]) * (x1 * theta[1] + theta[0])
    fig, ax = plt.subplots(figsize=(12, 8))
    groups = data.groupby('accepted')
    for name, group in groups:
        ax.scatter(group.exam1,
                   group.exam2,
                   s=50,
                   alpha=0.7,
                   label=labels[name])
    ax.plot(x1, x2)
    ax.legend(loc=1)
    ax.set_xlim([min(data.exam1) - 2, max(data.exam1) + 2])
    ax.set_ylim([min(data.exam2) - 2, max(data.exam2) + 2])
    ax.set_xlabel('Exam 1')
    ax.set_ylabel('Exam 2')
    ax.set_title('Scores of exams')
    plt.show()


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

result = opt.fmin_tnc(func=compute_cost, x0=theta, fprime=gradient_function, args=(X, y), disp=False)
theta = np.array(result[0]).reshape(3, 1)


p = sigmoid(np.dot(np.array([1, 45, 85]), theta))[0] * 100
print('For a student with scores 45 and 85, we predict an admission probability of: {}'.format(p))

predictions = predict(theta, X)
print('Training Accuracy: {}'.format(np.mean(predictions == y) * 100))


plot_decision_boundary(data)
