# Machine Learning Online Class - Lesson 2b: Regularised Logistic Classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

path = 'data/ex2data2.txt'
data = pd.read_csv(path, header=None, names=['test1', 'test2', 'accepted'])
labels = {0: 'Not Accepted', 1: 'Accepted'}


# fig, ax = plt.subplots(figsize=(8, 8))
# groups = data.groupby('accepted')
# for name, group in groups:
#     ax.scatter(group.test1,
#                group.test2,
#                s=50,
#                alpha=0.7,
#                label=labels[name])
# ax.legend(loc=1)
# ax.set_xlabel('Test 1 Score')
# ax.set_ylabel('Test 2 Score')
# ax.set_title('Results of tests')
# # plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, x, y, lmb):
    m = x.shape[0]
    h_x = sigmoid(np.dot(x, theta))
    theta_tmp = theta.copy()
    theta_tmp[0] = 0

    term_1 = -np.dot(y.T, np.log(h_x))
    term_2 = -np.dot((1 - y).T, np.log(1 - h_x))

    adj = lmb / 2 * np.dot(theta_tmp.T, theta_tmp)
    return 1 / m * (term_1 + term_2 + adj)


def gradient(theta, x, y, lmb):
    m = x.shape[0]
    h_x = sigmoid(np.dot(x, theta)).reshape(m, 1)
    theta_tmp = theta.copy()
    theta_tmp[0] = 0

    errors = h_x - y
    grad = 1 / m * np.dot(x.T, errors)
    adj = lmb / m * theta_tmp

    return grad.T + adj.T


def map_feature(df, power):
    for i in range(1, power + 1):
        for j in range(0, i + 1):
            df['F_{}{}'.format(i - j, j)] = np.power(df['test1'], i - j) * np.power(df['test2'], j)
    df.drop('test1', axis=1, inplace=True)
    df.drop('test2', axis=1, inplace=True)
    return df


def predict(x, theta):
    pred = sigmoid(np.dot(theta, x))
    pred = pred >= 0.5
    return pred


def plot_decision_boundary(data, th, degree):
    labels = {0: 'Not Accepted', 1: 'Accepted'}
    u = np.linspace(-1, 1.5, 20)
    v = np.linspace(-1, 1.5, 20)
    X_m, Y_m = np.meshgrid(u, v)
    Z = np.zeros([len(u), len(v)])
    for i, t0 in enumerate(u):
        for j, t1 in enumerate(v):
            df = pd.DataFrame({'test1': pd.Series(u[i]), 'test2': pd.Series(v[j])})
            df.insert(0, 'const', 1)
            df = map_feature(df, degree)
            df = np.array(df)
            # Need to transpose
            Z[j, i] = np.dot(df, th)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contour(X_m, Y_m, Z, levels=[0])
    groups = data.groupby('accepted')
    for name, group in groups:
        ax.scatter(group.test1,
                   group.test2,
                   s=50,
                   alpha=0.7,
                   label=labels[name])
    ax.legend(loc=1)
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')
    ax.set_title('Results of tests')
    plt.show()


# Set X and y
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]
X.insert(0, 'const', 1)

# Mapping additional features
degree = 6
X = map_feature(X, degree)

# Conversion to arrays
X = np.array(X)
y = np.array(y)

# Initialize theta and set hyper parameters
theta = np.zeros([X.shape[1], 1])
theta_reg_test = np.ones([X.shape[1], 1])
lmb = 1

# Cost of initial setup, no regularisation
print('No-regularisation, all thetas equal to zero => J = {:.2f} (Expected J = 0.69)'.
      format(np.float(cost_function(theta, X, y, 0))))
# Cost of test setup, with regularisation (lambda = 10)
print('Lambda = 10 regularisation, all thetas equal to ones => J = {:.2f} (Expected J = 3.16)\n'.
      format(np.float(cost_function(theta_reg_test, X, y, 10))))

# Finding optima theta
result = opt.fmin_tnc(func=cost_function, x0=theta, fprime=gradient, args=(X, y, lmb), disp=False)
print(result[0])
theta = np.matrix(result[0]).T

# Prediction accuracy
predictions = predict(theta, X)
print('\nTraining Accuracy: {}'.format(np.mean(predictions == y) * 100))

plot_decision_boundary(data, theta, degree)
