import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")

# Baseline 1: always predict the most common label in training
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit([[0]] * len(train), train["label"])
dummy_pred = dummy.predict([[0]] * len(val))
print("Baseline 1 (always guess most common):",
      round(accuracy_score(val["label"], dummy_pred), 3))

# Baseline 2: a hand-written keyword rule
positive_words = ["great", "love", "excellent", "good", "best", "perfect"]
negative_words = ["bad", "terrible", "waste", "poor", "worst", "disappointed"]

def keyword_rule(text):
    words = text.lower().split()
    pos = sum(w in positive_words for w in words)
    neg = sum(w in negative_words for w in words)
    return 1 if pos > neg else 0

rule_pred = val["text"].apply(keyword_rule)
print("Baseline 2 (keyword rule):",
      round(accuracy_score(val["label"], rule_pred), 3))