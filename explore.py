import pandas as pd

# Load the saved CSV file into a DataFrame
df = pd.read_csv("data/reviews_sample.csv")

# How big is it, and what are the columns?
print("Shape (rows, columns):", df.shape)
print("Column names:", list(df.columns))
print()

# Combine the title and the review body into one text column
df["text"] = df["title"].fillna("") + ". " + df["content"].fillna("")

# Is the data balanced?
print("Label counts:")
print(df["label"].value_counts())
print()

# Read three full reviews with their labels
for i in range(10):
    print("Label:", df["label"].iloc[i])
    print(df["text"].iloc[i])
    print("-----")