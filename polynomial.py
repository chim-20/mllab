import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('auto-mpg.csv', na_values='?')

df = df[['displacement', 'mpg']].dropna()

X = df[['displacement']].values
y = df['mpg'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

for degree in [2, 3, 4]:

    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_poly = model.predict(X_test_poly)

    print("\nPolynomial Degree:", degree)
    print("MSE:", mean_squared_error(y_test, y_poly))
    print("R2 Score:", r2_score(y_test, y_poly))

    idx = np.argsort(X_test.flatten())

    plt.scatter(X_test, y_test, color='blue', label='Actual data')

    plt.plot(X_test.flatten()[idx], y_poly[idx],
             label=f'Degree {degree}')

    plt.xlabel('Displacement')
    plt.ylabel('MPG')
    plt.title('Polynomial Regression')
    plt.legend()
    plt.show()
