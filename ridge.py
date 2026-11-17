import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes(as_frame=True)
df = data.frame

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

params = {'model__alpha': [0.01, 0.1, 1, 10, 100]}

grid_ridge = GridSearchCV(
    ridge, params, cv=5, scoring='neg_mean_squared_error'
)

grid_ridge.fit(X_train, y_train)

y_ridge = grid_ridge.predict(X_test)

print("\nRidge Regression")
print("Best Alpha:", grid_ridge.best_params_)
print("MSE:", mean_squared_error(y_test, y_ridge))
print("R2 Score:", r2_score(y_test, y_ridge))

lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso(max_iter=10000))
])

grid_lasso = GridSearchCV(
    lasso, params, cv=5, scoring='neg_mean_squared_error'
)

grid_lasso.fit(X_train, y_train)

y_lasso = grid_lasso.predict(X_test)

print("\nLasso Regression")
print("Best Alpha:", grid_lasso.best_params_)
print("MSE:", mean_squared_error(y_test, y_lasso))
print("R2 Score:", r2_score(y_test, y_lasso))
