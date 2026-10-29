import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Final settings, chosen using the validation set
vectorizer = TfidfVectorizer(max_features=50000, min_df=5, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train["text"])
X_test = vectorizer.transform(test["text"])

model = LogisticRegression(C=4, max_iter=1000)
model.fit(X_train, train["label"])
pred = model.predict(X_test)

cm = confusion_matrix(test["label"], pred)
print("Confusion matrix (TEST set)")
print("                 predicted neg   predicted pos")
print("actually neg     ", cm[0][0], "          ", cm[0][1])
print("actually pos     ", cm[1][0], "          ", cm[1][1])
print()
print(classification_report(
    test["label"], pred, target_names=["negative", "positive"], digits=3
))