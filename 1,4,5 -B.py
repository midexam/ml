# Linear Regression on Iris Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
# Load Iris Dataset
iris = load_iris()
# Taking one feature (Sepal Length)
X = iris.data[:, 0].reshape(-1, 1)
# Target variable
y = iris.data[:, 1]
# Create Model
model = LinearRegression()
# Train Model
model.fit(X, y)
# Prediction
y_pred = model.predict(X)
# Print Slope and Intercept
print("Slope(m):", model.coef_[0])
print("Intercept(b):", model.intercept_)
# Plot Graph
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear Regression on Iris Dataset")
plt.show()