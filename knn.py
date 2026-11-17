import time

k_values = [1, 3, 5, 7, 9]

results = []

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    start = time.time()

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    end = time.time()

    acc = accuracy_score(y_test, y_pred)

    results.append([k, acc, end - start])

    print("K =", k)
    print("Accuracy:", acc)
    print("Execution Time:", end - start)
    print("----------------------")

result_df = pd.DataFrame(
    results,
    columns=['K', 'Accuracy', 'Execution Time']
)

print("\nFinal Comparison:")
print(result_df)