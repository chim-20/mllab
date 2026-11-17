import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('fashion-mnist.csv')

X = df.drop('label', axis=1).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = X_train.reshape(len(X_train), -1) / 255.0
X_test = X_test.reshape(len(X_test), -1) / 255.0

k_values = [1, 3, 5, 7, 9]

results = []

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    start = time.time()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    end = time.time()

    acc = accuracy_score(y_test, y_pred)

    results.append([k, acc, end-start])

    print("K =", k)
    print("Accuracy:", acc)
    print("Execution Time:", end-start)
    print("--------------------")

result_df = pd.DataFrame(
    results,
    columns=['K', 'Accuracy', 'Execution Time']
)

print(result_df)
