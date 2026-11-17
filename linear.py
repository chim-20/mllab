import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing(as_frame=True)
df = data.frame

X = df[['AveRooms']].values
y = df['MedHouseVal'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

w = 0.0
b = 0.0
lr = 0.1
epochs = 1000
n = len(X_train)

for i in range(epochs):
    y_pred = w * X_train.flatten() + b

    dw = (2/n) * np.sum((y_pred - y_train) * X_train.flatten())
    db = (2/n) * np.sum(y_pred - y_train)

    w = w - lr * dw
    b = b - lr * db

y_gd = w * X_test.flatten() + b

X_b = np.c_[np.ones((len(X_train), 1)), X_train]

theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_train

b_ne = theta[0]
w_ne = theta[1]

y_ne = w_ne * X_test.flatten() + b_ne

print("Gradient Descent")
print("MSE:", mean_squared_error(y_test, y_gd))
print("R2 Score:", r2_score(y_test, y_gd))

print("\nNormal Equation")
print("MSE:", mean_squared_error(y_test, y_ne))
print("R2 Score:", r2_score(y_test, y_ne))

plt.scatter(X_test, y_test, color='blue', label='Actual data')

idx = np.argsort(X_test.flatten())

plt.plot(X_test.flatten()[idx], y_gd[idx],
         color='red', label='Gradient Descent')

plt.plot(X_test.flatten()[idx], y_ne[idx],
         color='green', linestyle='--', label='Normal Equation')

plt.xlabel('Average Rooms (Scaled)')
plt.ylabel('Median House Value')
plt.title('Linear Regression')
plt.legend()
plt.show()
