import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

train = pd.read_csv("data/train.csv")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50000, min_df=5, ngram_range=(1, 2))),
    ("model", LogisticRegression(C=4, max_iter=1000)),
])

# One command trains both steps, in order
pipeline.fit(train["text"], train["label"])

joblib.dump(pipeline, "models/sentiment_pipeline.joblib")
print("Pipeline saved to models/sentiment_pipeline.joblib")