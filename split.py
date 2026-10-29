import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/reviews_clean.csv")

# Step 1: set aside 20% of everything as the test set
train_val, test = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["label"]
)

# Step 2: from the remaining 80%, set aside 25% as validation
# (25% of 80% = 20% of the original data)
train, val = train_test_split(
    train_val, test_size=0.25, random_state=42, stratify=train_val["label"]
)

print("Train size:", len(train))
print("Validation size:", len(val))
print("Test size:", len(test))
print()

# Is each split still balanced?
print("Share of positive reviews:")
print("  train:", round(train["label"].mean(), 3))
print("  validation:", round(val["label"].mean(), 3))
print("  test:", round(test["label"].mean(), 3))
print()

# Leakage check: does any review text appear in more than one split?
def clean_keys(d):
    return set(d["text"].str.lower().str.strip())

print("Train & test overlap:", len(clean_keys(train) & clean_keys(test)))
print("Train & validation overlap:", len(clean_keys(train) & clean_keys(val)))
print("Validation & test overlap:", len(clean_keys(val) & clean_keys(test)))

# Save the three files
train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)
print()
print("Saved train.csv, val.csv, test.csv in the data folder")