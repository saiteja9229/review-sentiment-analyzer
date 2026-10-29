import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")

configs = [
	{"name": "words only,  C=1 ", "ngram": (1, 1), "C": 1},
	{"name": "words only,  C=4 ", "ngram": (1, 1), "C": 4},
	{"name": "words+pairs, C=1 ", "ngram": (1, 2), "C": 1},
	{"name": "words+pairs, C=4 ", "ngram": (1, 2), "C": 4},
	{"name": "words+pairs, C=10", "ngram": (1, 2), "C": 10},
]

for cfg in configs:
	vectorizer = TfidfVectorizer(
		max_features=50000, min_df=5, ngram_range=cfg["ngram"]
	)
	X_train = vectorizer.fit_transform(train["text"])
	X_val = vectorizer.transform(val["text"])

	model = LogisticRegression(C=cfg["C"], max_iter=1000)
	model.fit(X_train, train["label"])

	train_acc = accuracy_score(train["label"], model.predict(X_train))
	val_acc = accuracy_score(val["label"], model.predict(X_val))
	print(
		cfg["name"],
		"| train:",
		round(train_acc, 3),
		"| validation:",
		round(val_acc, 3),
	)
