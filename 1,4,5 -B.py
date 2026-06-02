import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
iris = load_iris()
X = iris.data[:, 0].reshape(-1, 1)
y = iris.data[:, 1]
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
print("Slope(m):", model.coef_[0])
print("Intercept(b):", model.intercept_)
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear Regression on Iris Dataset")
plt.show()
