import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

df = pd.read_csv('20newsgroups.csv')

df = df.dropna(subset=['text', 'category'])

X = df['text'].astype(str)
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = CountVectorizer(stop_words='english', max_features=10000)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Multinomial Naive Bayes
mn = MultinomialNB()

mn.fit(X_train, y_train)

y_mn = mn.predict(X_test)

acc_mn = accuracy_score(y_test, y_mn)
f1_mn = f1_score(y_test, y_mn, average='weighted')

print("Multinomial Naive Bayes")
print("Accuracy score:", acc_mn)
print("F1 score:", f1_mn)

bnb = BernoulliNB()

bnb.fit(X_train, y_train)

y_bnb = bnb.predict(X_test)

acc_bnb = accuracy_score(y_test, y_bnb)
f1_bnb = f1_score(y_test, y_bnb, average='weighted')

print("\nBernoulli Naive Bayes")
print("Accuracy score:", acc_bnb)
print("F1 score:", f1_bnb)
print("\nModel Comparison")

print("Multinomial NB:")
print("Accuracy:", acc_mn)
print("F1 Score:", f1_mn)

print("\nBernoulli NB:")
print("Accuracy:", acc_bnb)
print("F1 Score:", f1_bnb)