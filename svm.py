
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data[:, 2:4]
y = (iris.target != 0).astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = SVC(kernel='linear', C=1)
model.fit(X, y)

y_pred = model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))

plt.scatter(X[:, 0], X[:, 1], c=y)

w = model.coef_[0]
b = model.intercept_[0]

x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

boundary = -(w[0] * x + b) / w[1]
margin1 = -(w[0] * x + b - 1) / w[1]
margin2 = -(w[0] * x + b + 1) / w[1]

plt.plot(x, boundary, 'k-')
plt.plot(x, margin1, 'k--')
plt.plot(x, margin2, 'k--')

plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=100, facecolors='none', edgecolors='red'
)

plt.xlabel('Petal Length (scaled)')
plt.ylabel('Petal Width (scaled)')
plt.title('Linear SVM')
plt.show()
