import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")

# Text -> numbers (vocabulary learned from training only)
vectorizer = TfidfVectorizer(max_features=20000, min_df=5)
X_train = vectorizer.fit_transform(train["text"])
X_val = vectorizer.transform(val["text"])

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train["label"])

# Check accuracy on both sets
train_acc = accuracy_score(train["label"], model.predict(X_train))
val_acc = accuracy_score(val["label"], model.predict(X_val))
print("Training accuracy:", round(train_acc, 3))
print("Validation accuracy:", round(val_acc, 3))
print()

# Which words push hardest toward each label?
vocab = vectorizer.get_feature_names_out()
weights = model.coef_[0]
order = np.argsort(weights)

print("Top 10 NEGATIVE words:", [vocab[i] for i in order[:10]])
print("Top 10 POSITIVE words:", [vocab[i] for i in order[::-1][:10]])
