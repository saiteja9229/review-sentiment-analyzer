from datasets import load_dataset


# Download the training portion of the dataset.
dataset = load_dataset("fancyzhx/amazon_polarity", split="train")
print("Total reviews available:", len(dataset))

# Shuffle randomly, then keep the first 50,000 reviews.
sample = dataset.shuffle(seed=42).select(range(50000))

# Convert to a pandas table and save as a CSV file.
df = sample.to_pandas()
df.to_csv("data/reviews_sample.csv", index=False)

print("Saved", len(df), "reviews to data/reviews_sample.csv")
print(df.head(3))
