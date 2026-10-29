import joblib
import streamlit as st

st.set_page_config(page_title="Review Sentiment Analyzer")


@st.cache_resource
def load_model():
    return joblib.load("models/sentiment_pipeline.joblib")


pipeline = load_model()

st.title("Customer Review Sentiment Analyzer")
st.write(
    "Type or paste an English-language product review. The model will "
    "predict whether it sounds positive or negative."
)

review = st.text_area("Your review", height=150)

if st.button("Analyze"):
    if not review.strip():
        st.warning("Please type or paste a review first.")
    else:
        prob_positive = pipeline.predict_proba([review])[0][1]

        if prob_positive >= 0.65:
            st.success("Positive")
            st.write(f"Model confidence: {prob_positive:.0%}")
        elif prob_positive <= 0.35:
            st.error("Negative")
            st.write(f"Model confidence: {1 - prob_positive:.0%}")
        else:
            st.info(
                "Uncertain. This review may be mixed or neutral, "
                "and the model is not confident either way."
            )
            st.write(f"Probability of positive: {prob_positive:.0%}")

st.caption(
    "Limitations: this model only knows two labels (positive and negative), "
    "so it cannot reliably recognize neutral reviews. It was trained on "
    "English Amazon product reviews and may be less accurate on other "
    "industries, slang, sarcasm, or other languages."
)