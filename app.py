import streamlit as st
import joblib
import re
import math
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

# Get the folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "sentiment_svm.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"


# Check whether required files exist
if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not VECTORIZER_PATH.exists():
    st.error(f"TF-IDF vectorizer not found: {VECTORIZER_PATH}")
    st.stop()


# Load saved model and vectorizer
try:
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean movie review text in the same basic way
    used during the project preprocessing.
    """

    # Remove HTML line break tags such as <br /> and <br/>
    text = re.sub(r"<br\s*/?>", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# HEADER
# ============================================================

st.title("🎬 Movie Sentiment Analyzer")

st.write(
    "Enter a movie review and our machine learning model "
    "will classify it as **positive** or **negative**."
)

st.markdown("---")


# ============================================================
# REVIEW INPUT
# ============================================================

st.subheader("📝 Enter Your Review")

review = st.text_area(
    "Enter your movie review:",
    height=180,
    placeholder=(
        "Example: This movie was fantastic! "
        "The acting was brilliant and I really enjoyed it."
    )
)


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if not review.strip():
        st.warning("Please enter a movie review first.")

    else:

        # Clean the review
        cleaned_review = clean_text(review)

        # Convert text into TF-IDF features
        review_tfidf = tfidf.transform([cleaned_review])

        # Predict sentiment
        prediction = model.predict(review_tfidf)[0]

        # Get SVM decision score
        decision_score = model.decision_function(review_tfidf)[0]

        # ----------------------------------------------------
        # Convert decision score into a confidence-style value
        # ----------------------------------------------------

        # This is NOT a calibrated probability.
        # It is only a confidence-style indicator based
        # on the distance from the SVM decision boundary.

        confidence = (
            1 / (1 + math.exp(-abs(float(decision_score))))
        ) * 100

        confidence = min(confidence, 99.9)

        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        if prediction == "positive":

            st.success("😊 POSITIVE REVIEW")

        else:

            st.error("😞 NEGATIVE REVIEW")

        # ----------------------------------------------------
        # Display confidence
        # ----------------------------------------------------

        st.subheader("Model Confidence")

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.1f}%"
        )

        st.caption(
            "Confidence is derived from the SVM decision score "
            "and is not a calibrated probability."
        )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown("---")

st.header("🧠 How It Works")

col1, col2, col3 = st.columns(3)


with col1:

    st.subheader("1️⃣ Text")

    st.write(
        "The movie review is cleaned and prepared "
        "for analysis."
    )


with col2:

    st.subheader("2️⃣ TF-IDF")

    st.write(
        "Important words and phrases are converted "
        "into numerical features using TF-IDF."
    )


with col3:

    st.subheader("3️⃣ SVM")

    st.write(
        "The trained Linear SVM classifies the review "
        "as positive or negative."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.header("📊 Model Performance")


# Top metrics

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Final Model",
        "Linear SVM"
    )


with col2:

    st.metric(
        "Test Accuracy",
        "91.07%"
    )


with col3:

    st.metric(
        "Test Reviews",
        "9,917"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader("🔬 Models Tested")

st.write(
    "Four machine learning classification models were "
    "evaluated using the same test dataset:"
)


comparison = {
    "Logistic Regression": "90.43%",
    "Naive Bayes": "88.29%",
    "Random Forest": "85.47%",
    "Linear SVM": "91.07%"
}


for model_name, accuracy in comparison.items():

    st.write(
        f"**{model_name}** — {accuracy}"
    )


# ============================================================
# DATASET
# ============================================================

st.markdown("---")

st.header("📚 Dataset")

st.write(
    "The model was trained using the IMDB Movie Reviews "
    "dataset containing 50,000 labeled reviews."
)

col1, col2 = st.columns(2)


with col1:

    st.markdown("### Dataset Details")

    st.write("• 25,000 positive reviews")
    st.write("• 25,000 negative reviews")
    st.write("• 80% training data")
    st.write("• 20% testing data")


with col2:

    st.markdown("### Final Dataset")

    st.write("• 49,582 reviews after duplicate removal")
    st.write("• 39,665 training reviews")
    st.write("• 9,917 testing reviews")
    st.write("• 50,000 TF-IDF features")


# ============================================================
# ERROR ANALYSIS
# ============================================================

st.markdown("---")

st.header("🔎 Error Analysis")

st.write(
    "The final Linear SVM was evaluated on 9,917 test reviews."
)

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Incorrect Predictions",
        "886"
    )


with col2:

    st.metric(
        "Error Rate",
        "8.93%"
    )


with col3:

    st.metric(
        "Correct Predictions",
        "9,031"
    )


st.write(
    "The model made 484 negative-to-positive errors "
    "and 402 positive-to-negative errors."
)


# ============================================================
# TECHNOLOGY STACK
# ============================================================

st.markdown("---")

st.header("🛠️ Technology Stack")

tech_col1, tech_col2 = st.columns(2)


with tech_col1:

    st.markdown("### Development")

    st.write("🐍 Python")
    st.write("🐼 Pandas")
    st.write("🔢 NumPy")
    st.write("📊 Matplotlib")
    st.write("📈 Seaborn")


with tech_col2:

    st.markdown("### Machine Learning")

    st.write("🧠 Scikit-learn")
    st.write("📝 TF-IDF Vectorizer")
    st.write("⚡ Linear SVM")
    st.write("💾 Joblib")
    st.write("🌐 Streamlit")


# ============================================================
# CONCLUSION
# ============================================================

st.markdown("---")

st.header("✅ Conclusion")

st.write(
    "The Movie Sentiment Analysis system uses Natural Language "
    "Processing and machine learning to classify movie reviews "
    "as positive or negative."
)

st.write(
    "After preprocessing the IMDB dataset, extracting TF-IDF "
    "features, comparing multiple machine learning models, "
    "and tuning the Linear SVM, the final model achieved "
    "91.07% accuracy on the test dataset."
)

st.write(
    "The trained model was integrated into a Streamlit "
    "application that allows users to enter a movie review "
    "and receive an automatic sentiment prediction."
)


# ============================================================
# FUTURE SCOPE
# ============================================================

st.header("🚀 Future Scope")

st.markdown("""
- Support **neutral and mixed sentiments**.
- Experiment with advanced NLP techniques.
- Explore **Word2Vec, GloVe and transformer-based models**.
- Support **multilingual sentiment analysis**.
- Deploy the application as a cloud-based service.
- Provide more detailed explanations for predictions.
""")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Movie Sentiment Analysis | IMDB Dataset | "
    "Linear SVM + TF-IDF"
)