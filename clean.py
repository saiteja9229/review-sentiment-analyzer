import pandas as pd

df = pd.read_csv("data/reviews_sample.csv")
df["text"] = df["title"].fillna("") + ". " + df["content"].fillna("")

# A "key" ignores capital letters and extra spaces, so
# "Great!" and "great! " count as the same review
df["key"] = df["text"].str.lower().str.strip()

print("Rows before:", len(df))

# Count extra copies of reviews we've already seen
n_dupes = df.duplicated(subset="key").sum()
print("Duplicate rows (extra copies):", n_dupes)

# Same text but different labels?
label_counts = df.groupby("key")["label"].nunique()
n_conflict = (label_counts > 1).sum()
print("Review texts with conflicting labels:", n_conflict)

# Show up to 6 rows involved in duplication
dupes = df[df.duplicated(subset="key", keep=False)].sort_values("key")
preview = dupes[["label", "text"]].head(6).copy()
preview["text"] = preview["text"].str[:80]
print(preview)

# Remove: first drop texts with conflicting labels, then keep one copy of each
conflict_keys = label_counts[label_counts > 1].index
df = df[~df["key"].isin(conflict_keys)]
df = df.drop_duplicates(subset="key", keep="first")

print("Rows after:", len(df))
print(df["label"].value_counts())

# Save the clean version for the next steps
df[["label", "text"]].to_csv("data/reviews_clean.csv", index=False)
print("Saved data/reviews_clean.csv")