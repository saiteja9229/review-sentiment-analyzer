import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")

vectorizer = TfidfVectorizer(max_features=20000, min_df=5)
X_train = vectorizer.fit_transform(train["text"])
X_val = vectorizer.transform(val["text"])

model = LogisticRegression(max_iter=1000)
model.fit(X_train, train["label"])
pred = model.predict(X_val)

# The confusion matrix: rows = truth, columns = model's guess
cm = confusion_matrix(val["label"], pred)
print("Confusion matrix (validation set)")
print("                 predicted neg   predicted pos")
print("actually neg     ", cm[0][0], "          ", cm[0][1])
print("actually pos     ", cm[1][0], "          ", cm[1][1])
print()

# Precision, recall, F1 for each class
print(classification_report(
    val["label"], pred, target_names=["negative", "positive"], digits=3
))