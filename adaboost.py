import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

data = pd.read_csv('bupa.data', header = None)
data.to_csv('bupa.csv', index = False, header = False)
columns = ["mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks", "selector"]
data.columns = columns
print(data.head())

data['label'] = data['drinks'].apply(lambda x: 1 if x > 3 else -1)
print(data['label'].value_counts())
X = data.drop(columns=['drinks', 'selector', 'label'])
y = data['label'].values


X = (X - X.mean()) / X.std()

class DecisionStump:

    def __init__(self):
        self.j = None
        self.c = None
        self.C1 = None
        self.C2 = None

    def fit(self, X, y, D):

        m, n = X.shape
        best_error = float('inf')

        for j in range(n):
            thresholds = np.unique(X[:, j])
            for threshold in thresholds:
                for C1, C2 in [(1, -1), (-1, 1)]:
                    pred = np.where(X[:, j] >= threshold, C1, C2)
                    error = np.sum(D[pred] != y)
                    if error < best_error:
                        best_error = error
                        self.j, self.c, self.C1, self.C2 = j, threshold, C1, C2
    
    def predict(self, X):
        return np.where(X[:, self.j] >= self.c, self.C1, self.C2)

class AdaBoost:
    
    def __init__(self, T = 100):
        self.T = T
        self.alphas = []
        self.stumps = []

    def fit(self, X, y):
        m = len(y)
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        D = np.ones(m) / m
        D[y == 1] = class_weights[1]
        D[y == -1] = class_weights[0]
        D = D / np.sum(D)

        for t in range(self.T):
            stump = DecisionStump()
            stump.fit(X, y, D)
            pred = stump.predict(X)
            error = np.sum(D[pred != y])
            alpha = 0.5 * np.log((1 - error) / error)
            D = D * np.exp(-alpha * y * pred)
            D = D / np.sum(D)

            self.stumps.append(stump)
            self.alphas.append(alpha)

            if t < 10:
                print(f"Iteration {t+1}: Feature j = {stump.j}, Threshold c = {stump.c}, Class label C1 = {stump.C1}")

    def predict(self, X):
        H = sum(alpha * stump.predict(X) for alpha, stump in zip(self.alphas, self.stumps))
        return np.sign(H)      
    
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.1)

adaboost = AdaBoost(T = 100)
adaboost.fit(X_train, y_train)

train_pred = adaboost.predict(X_train)
test_pred = adaboost.predict(X_test)
print("train accuracy: ", accuracy_score(y_train, train_pred))
print("test accuracy: ", accuracy_score(y_test, test_pred))

train_errors = np.zeros(100)
test_errors = np.zeros(100)

for _ in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.1)
    adaboost = AdaBoost(T=100)
    adaboost.fit(X_train, y_train)
    
    for t in range(100):
        train_pred = adaboost.predict(X_train)
        test_pred = adaboost.predict(X_test)
        train_errors[t] += 1 - accuracy_score(y_train, train_pred)
        test_errors[t] += 1 - accuracy_score(y_test, test_pred)

train_errors /= 50
test_errors /= 50

plt.plot(train_errors, label='Train Error')
plt.plot(test_errors, label='Test Error')
plt.xlabel('Boosting Iterations')
plt.ylabel('Error Rate')
plt.legend()
plt.show()