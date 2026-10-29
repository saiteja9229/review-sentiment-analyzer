import joblib

pipeline = joblib.load("models/sentiment_pipeline.joblib")

reviews = [
    "Absolutely love this blender, it works perfectly!",
    "Terrible quality. It broke after two days. Waste of money.",
    "The battery is great but the screen is awful.",
    "It was okay.",
]

labels = {0: "negative", 1: "positive"}

for text in reviews:
    prob_positive = pipeline.predict_proba([text])[0][1]
    label = labels[int(prob_positive >= 0.5)]
    print(label, "| positive probability:", round(prob_positive, 2), "|", text)