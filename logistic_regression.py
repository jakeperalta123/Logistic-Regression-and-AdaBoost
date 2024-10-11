import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

import sklearn.linear_model as skl_lm
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split


import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv('breast_cancer.data', header=None, delim_whitespace=True)

data.to_csv('breast_cancer.csv', index=False, header=False)

X = data.iloc[:, 1:10].values
y = data.iloc[:, 9].values

y = np.where(y == 2,0,1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, (h- y ) / m)
        theta -= learning_rate * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

def predict(X, theta):
    return sigmoid(np.dot(X, theta)) >= 0.5

def acc_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy

learning_rate = 0.1
iterations = 1000
n_features = X_scaled.shape[1]
theta_init = np.zeros(n_features)

fractions = [0.01, 0.02, 0.03, 0.125, 0.625, 1.0]
accuracies = []

for frac in fractions:
    acc_list = []
    for _ in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1/3, random_state=None)
        
        X_train_frac = X_train[:int(frac * len(X_train))]
        y_train_frac = y_train[:int(frac * len(y_train))]
        
        theta, _ = gradient_descent(X_train_frac, y_train_frac, theta_init.copy(), learning_rate, iterations)
        
        y_pred = predict(X_test, theta)
        acc = acc_score(y_test, y_pred)
        acc_list.append(acc)
    
    accuracies.append(np.mean(acc_list))

plt.plot(fractions, accuracies, marker='o')
plt.title('Learning Curve: Accuracy vs. Training Data Size')
plt.xlabel('Fraction of Training Data')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
