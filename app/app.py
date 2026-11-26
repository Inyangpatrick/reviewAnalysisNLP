
import streamlit as st
import joblib as jbl
import pandas as pd
import re
import string

# @st.cache_resource
model = jbl.load("../model/model.joblib")
vectorizer = jbl.load("../model/vectorizer.joblib")


st.set_page_config(page_title="Review Sentiment Analysis",layout="centered",initial_sidebar_state="expanded")

st.title("Review Sentiment Analysis")
st.write("Enter a review or upload a CSV file to predict sentiment.")

def clean_review(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\bbr\b", " ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_sentiment(reviews):
    cleaned = [clean_review(review) for review in reviews]
    vectors = vectorizer.transform(cleaned)

    predictions = model.predict(vectors)
    probabilities = model.predict_proba(vectors)[:, 1]

    results = pd.DataFrame({
        "Review": reviews,
        "Predicted Sentiment": [
            "Positive" if prediction == 1 else "Negative"
            for prediction in predictions
        ],
        "Positive Probability": probabilities
    })

    return results

option = st.radio(
    "Choose input method:",
    ["Enter Review Manually", "Upload CSV"],
    horizontal=True
)

#Manual
if option == "Enter Review Manually":

    review = st.text_area("Enter Review:")

    if st.button("Analyze Review"):

        if review.strip():

            results = predict_sentiment([review])

            sentiment = results["Predicted Sentiment"][0]
            probability = results["Positive Probability"][0]

            if sentiment == "Positive":
                st.success("Positive Review")
            else:
                st.error("Negative Review")

            st.write(f"Positive Probability: {probability:.2%}")

        else:
            st.warning("Please enter a movie review.")

# CSV 
else:

    uploaded_file = st.file_uploader(
        "Upload CSV containing reviews",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("Dataset Preview")
        st.dataframe(df.head())

        if "review" not in df.columns:
            st.error("CSV must contain a column named 'review'.")

        else:

            if st.button("Analyze CSV"):

                valid_reviews = df["review"].dropna().astype(str)

                if len(valid_reviews) == 0:
                    st.warning("No valid reviews found.")

                else:
                    results = predict_sentiment(valid_reviews.tolist())

                    st.success(
                        f"Successfully analyzed {len(results)} reviews."
                    )

                    st.dataframe(results, use_container_width=True)

                    # Download predictions
                    csv_data = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Predictions",
                        data=csv_data,
                        file_name="sentiment_predictions.csv",
                        mime="text/csv"
                    )




    