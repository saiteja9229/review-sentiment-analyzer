import joblib
import pandas as pd

pipeline = joblib.load("models/sentiment_pipeline.joblib")
val = pd.read_csv("data/val.csv")

val["prob_pos"] = pipeline.predict_proba(val["text"])[:, 1]
val["pred"] = (val["prob_pos"] >= 0.5).astype(int)
val["correct"] = val["pred"] == val["label"]

uncertain = val["prob_pos"].between(0.35, 0.65)

print("Share of reviews in the uncertain band:", round(uncertain.mean(), 3))
print("Accuracy inside the uncertain band:", round(val.loc[uncertain, "correct"].mean(), 3))
print("Accuracy outside the band:", round(val.loc[~uncertain, "correct"].mean(), 3))