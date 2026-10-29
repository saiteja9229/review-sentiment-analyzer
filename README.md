# Customer Review Sentiment Analyzer

A web app that predicts whether a product review is positive or negative,
built with Python, scikit-learn, and Streamlit.

**Live demo:** (link will be added after deployment)

## Why this matters

Businesses receive far more reviews than anyone can read. A sentiment model
can help teams track customer satisfaction over time, spot sudden increases in
complaints, and route unhappy customers to support faster. The app also marks
low-confidence predictions as "uncertain" so a person can review them.

## Data

- **Dataset:** Amazon Review Polarity, a labeled set of English Amazon product
	reviews (original reviews collected by McAuley and Leskovec, 2013; polarity
	version prepared by Xiang Zhang and hosted on Hugging Face).
- **Labels:** 1-2 star reviews are negative, 4-5 star reviews are positive.
	3-star reviews are not included.
- **License:** listed as Apache 2.0 on the Hugging Face dataset card. The
	underlying reviews were collected for academic research. This project is for
	learning and portfolio use.
- **Sample used:** 50,000 randomly selected reviews, with no duplicate or
	conflicting reviews found.

## Method

1. Split the data 60/20/20 into train (30,000), validation (10,000), and test
	 (10,000) sets, and checked that no review appears in more than one split.
2. Converted text to numbers with TF-IDF (single words and two-word pairs).
3. Trained a logistic regression classifier.
4. Chose settings using the validation set only.
5. Scored the final model once on the untouched test set.
6. Saved the vectorizer and model together as one scikit-learn pipeline.

## Results

| Model | Validation accuracy |
|---|---|
| Always guess the most common label | 50.1% |
| Hand-written keyword rule | 69.8% |
| Logistic regression, single words | 87.9% |
| Logistic regression, words + pairs (final) | 90.6% |

**Final test set:** 90.4% accuracy, with precision, recall, and F1 of about
0.90 for both classes.

**Uncertainty band:** on the validation set, predictions with a positive
probability between 35% and 65% (12.2% of reviews) were correct 62.2% of the
time, compared with 94.6% for all other predictions.

## Limitations

- **Binary only:** the model has two labels, so it cannot reliably recognize
	neutral reviews. The "uncertain" label helps with mixed reviews, but not
	with neutral text the model is confidently wrong about.
- **Domain:** trained on Amazon product reviews. It may be less accurate on
	restaurants, hotels, or other industries, because the common words and
	complaints differ.
- **Language:** English only. Other languages are not supported.
- **Meaning:** the model counts words and word pairs, so it can miss sarcasm,
	mixed opinions, and negative words about a competitor's product.
- **Labels:** labels come from star ratings, and some look wrong, so measured
	accuracy is only approximate.

## Run it yourself

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To rebuild the model from scratch you also need `pip install pandas datasets`,
then run these scripts in order: `download_data.py`, `clean.py`, `split.py`,
`train_pipeline.py`.

## Author

Sai Teja Raparthy
