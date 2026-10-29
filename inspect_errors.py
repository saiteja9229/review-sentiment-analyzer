import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")

vectorizer = TfidfVectorizer(max_features=20000, min_df=5)
X_train = vectorizer.fit_transform(train["text"])
X_val = vectorizer.transform(val["text"])

model = LogisticRegression(max_iter=1000)
model.fit(X_train, train["label"])

# Predictions and the model's probability that each review is positive
val["pred"] = model.predict(X_val)
val["prob_pos"] = model.predict_proba(X_val)[:, 1]

# Keep only the mistakes
errors = val[val["label"] != val["pred"]].copy()
print("Number of mistakes:", len(errors))
print()

# How sure was the model about its WRONG answer?
errors["confidence"] = errors["prob_pos"].where(
    errors["pred"] == 1, 1 - errors["prob_pos"]
)

names = {0: "negative", 1: "positive"}

def show(rows):
    for _, row in rows.iterrows():
        print("TRUE:", names[row["label"]], "| MODEL SAID:", names[row["pred"]],
              "| confidence:", round(row["confidence"], 2))
        print(row["text"][:300])
        print("-----")

print("=== 8 RANDOM MISTAKES ===")
show(errors.sample(8, random_state=1))

print()
print("=== 4 MOST CONFIDENT MISTAKES ===")
show(errors.sort_values("confidence", ascending=False).head(4))

# Save all mistakes so you can browse them in VS Code
errors.to_csv("data/val_errors.csv", index=False)