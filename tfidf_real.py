import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")

vectorizer = TfidfVectorizer(max_features=20000, min_df=5)

# Learn vocabulary from TRAINING reviews only, then convert them
X_train = vectorizer.fit_transform(train["text"])

# Convert validation reviews using the same vocabulary (no learning)
X_val = vectorizer.transform(val["text"])

print("Train matrix shape:", X_train.shape)
print("Validation matrix shape:", X_val.shape)

vocab = vectorizer.get_feature_names_out()
print("Vocabulary size:", len(vocab))
print("First 15 words:", list(vocab[:15]))
print()

# Which words matter most in the very first training review?
print("Review:", train["text"].iloc[0][:200])
row = X_train[0].toarray()[0]
top_positions = np.argsort(row)[::-1][:5]
for i in top_positions:
    print(vocab[i], round(row[i], 3))