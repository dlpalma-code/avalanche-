"""
Avalanche Product Review Sentiment Dashboard
---------------------------------------------
A simple Streamlit MVP for exploring product review sentiment.

Run with:
    streamlit run app.py
"""

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from a local .env file (for the optional LLM feature)
load_dotenv()

# ---------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Avalanche Sentiment Analysis",
    page_icon="📊",
    layout="wide",
)

REQUIRED_COLUMNS = ["PRODUCT", "DATE", "SUMMARY", "SENTIMENT_SCORE"]
DEFAULT_CSV_PATH = "avalanche_reviews.csv"

# ---------------------------------------------------------------------------
# SENTIMENT CATEGORY THRESHOLDS
# ---------------------------------------------------------------------------
# ASSUMPTION: SENTIMENT_SCORE is assumed to range roughly from -1 (very
# negative) to 1 (very positive), which is a common scale for sentiment
# scores produced by NLP models (e.g. VADER, TextBlob).
#
# If your dataset uses a different scale (e.g. 1-5 stars, or 0-100), change
# the two numbers below. Everything else in the app will update automatically.
POSITIVE_THRESHOLD = 0.05   # scores >= this value are "Positive"
NEGATIVE_THRESHOLD = -0.05  # scores <= this value are "Negative"
# Anything in between is considered "Neutral".


def categorize_sentiment(score: float) -> str:
    """Map a numeric sentiment score to a simple category label."""
    if score >= POSITIVE_THRESHOLD:
        return "Positive"
    elif score <= NEGATIVE_THRESHOLD:
        return "Negative"
    else:
        return "Neutral"


# ---------------------------------------------------------------------------
# 3. DATASET LOADING
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_or_path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame. Raises on failure so caller can handle it."""
    return pd.read_csv(file_or_path)


def get_raw_dataframe():
    """
    Try to load data from a user-uploaded file first.
    Fall back to a local default CSV if no file was uploaded.
    Returns (dataframe, source_description) or (None, None) on failure.
    """
    uploaded_file = st.sidebar.file_uploader("Upload a review CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = load_csv(uploaded_file)
            return df, "uploaded file"
        except Exception as e:
            st.error(f"Could not read the uploaded CSV file: {e}")
            return None, None

    # No file uploaded -> try the default local file
    if os.path.exists(DEFAULT_CSV_PATH):
        try:
            df = load_csv(DEFAULT_CSV_PATH)
            return df, f"default file ({DEFAULT_CSV_PATH})"
        except Exception as e:
            st.error(f"Could not read '{DEFAULT_CSV_PATH}': {e}")
            return None, None

    return None, None


# ---------------------------------------------------------------------------
# 4 & 5. DATA VALIDATION AND CLEANING
# ---------------------------------------------------------------------------
def validate_columns(df: pd.DataFrame):
    """Return a list of any required columns missing from df."""
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def clean_data(df: pd.DataFrame):
    """
    Clean the raw dataframe:
      - convert DATE to datetime (invalid dates become NaT, rows kept but flagged)
      - convert SENTIMENT_SCORE to numeric (invalid rows dropped)
      - drop rows with missing PRODUCT or SUMMARY
    Returns (clean_df, n_dropped)
    """
    df = df.copy()
    original_count = len(df)

    # Convert DATE - invalid dates become NaT rather than crashing the app
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    # Convert SENTIMENT_SCORE to numeric - invalid values become NaN
    df["SENTIMENT_SCORE"] = pd.to_numeric(df["SENTIMENT_SCORE"], errors="coerce")

    # Drop rows where sentiment score could not be interpreted
    df = df.dropna(subset=["SENTIMENT_SCORE"])

    # Drop rows missing essential text fields
    df = df.dropna(subset=["PRODUCT", "SUMMARY"])

    # Add a sentiment category column for easy filtering/charting
    df["SENTIMENT_CATEGORY"] = df["SENTIMENT_SCORE"].apply(categorize_sentiment)

    n_dropped = original_count - len(df)
    return df, n_dropped


# ---------------------------------------------------------------------------
# 11/13. NATURAL-LANGUAGE QUESTION HANDLING (Pandas-first, LLM optional)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a data-analysis assistant.

Answer questions using only the information provided from the customer-review dataset.

Do not invent statistics, products, reviews, or trends.

When numerical information is available, use the calculated values provided by the application.

If the dataset does not contain enough information to answer the question, clearly state that the information is unavailable.

Keep answers concise and explain calculations when useful."""


def try_answer_with_pandas(question: str, df: pd.DataFrame):
    """
    Attempt to answer common questions directly with Pandas.
    Returns an answer string, or None if the question isn't recognized.
    """
    if df.empty:
        return "There is no data available to answer that question (the current filters return zero rows)."

    q = question.lower()

    avg_by_product = df.groupby("PRODUCT")["SENTIMENT_SCORE"].mean()

    if "average" in q and "sentiment" in q and "product" not in q:
        return f"The average sentiment score is **{df['SENTIMENT_SCORE'].mean():.3f}** across {len(df)} reviews."

    if "how many" in q and "review" in q:
        return f"There are **{len(df)}** reviews in the current selection."

    if "highest" in q and "product" in q:
        best = avg_by_product.idxmax()
        return f"**{best}** has the highest average sentiment score ({avg_by_product.max():.3f})."

    if "lowest" in q and "product" in q:
        worst = avg_by_product.idxmin()
        return f"**{worst}** has the lowest average sentiment score ({avg_by_product.min():.3f})."

    if "most positive" in q:
        top = df.sort_values("SENTIMENT_SCORE", ascending=False).head(5)
        lines = "\n".join(f"- **{r.PRODUCT}** ({r.SENTIMENT_SCORE:.2f}): {r.SUMMARY}" for r in top.itertuples())
        return f"Top 5 most positive reviews:\n\n{lines}"

    if "most negative" in q:
        bottom = df.sort_values("SENTIMENT_SCORE", ascending=True).head(5)
        lines = "\n".join(f"- **{r.PRODUCT}** ({r.SENTIMENT_SCORE:.2f}): {r.SUMMARY}" for r in bottom.itertuples())
        return f"Top 5 most negative reviews:\n\n{lines}"

    if "median" in q and "sentiment" in q:
        return f"The median sentiment score is **{df['SENTIMENT_SCORE'].median():.3f}**."

    return None  # Not recognized -> caller may try the LLM


def build_data_summary(df: pd.DataFrame) -> str:
    """
    Build a small, safe summary of the dataset to send to the LLM instead of
    the raw data. Keeps the API call cheap and avoids leaking unnecessary data.
    """
    avg_by_product = df.groupby("PRODUCT")["SENTIMENT_SCORE"].mean().round(3).to_dict()
    summary = {
        "total_reviews": len(df),
        "average_sentiment": round(df["SENTIMENT_SCORE"].mean(), 3) if not df.empty else None,
        "min_sentiment": round(df["SENTIMENT_SCORE"].min(), 3) if not df.empty else None,
        "max_sentiment": round(df["SENTIMENT_SCORE"].max(), 3) if not df.empty else None,
        "median_sentiment": round(df["SENTIMENT_SCORE"].median(), 3) if not df.empty else None,
        "products": sorted(df["PRODUCT"].unique().tolist()),
        "average_sentiment_by_product": avg_by_product,
        "sentiment_category_counts": df["SENTIMENT_CATEGORY"].value_counts().to_dict(),
    }
    return str(summary)


# Hugging Face model used for the optional free-form question fallback.
# Any instruction-tuned chat model on the Hugging Face Inference API works here.
HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"


def get_api_key():
    """
    Look for the Hugging Face access token in two places, in order:
      1. Streamlit Cloud's secrets manager (st.secrets) - used in deployment
      2. A local environment variable / .env file - used for local development
    Returns None if it isn't set in either place.
    """
    try:
        # st.secrets raises if no secrets.toml exists at all (e.g. local dev
        # with only a .env file), so this is wrapped in a try/except.
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return os.getenv("HF_TOKEN")


def ask_llm(question: str, df: pd.DataFrame):
    """
    Optional LLM fallback. Only called if a Pandas-based answer wasn't found
    AND an API token is configured. Sends only a small data summary, never the
    full dataset.
    """
    api_key = get_api_key()
    if not api_key:
        return (
            "I couldn't answer that with a direct calculation, and no Hugging Face "
            "token is configured. Add HF_TOKEN to your .env file to enable free-form "
            "questions, or try rephrasing your question."
        )

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=api_key)
        data_summary = build_data_summary(df)

        response = client.chat_completion(
            model=HF_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Dataset summary:\n{data_summary}\n\nQuestion: {question}",
                },
            ],
            max_tokens=300,
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"The LLM request failed: {e}"


# ---------------------------------------------------------------------------
# APP HEADER
# ---------------------------------------------------------------------------
st.title("Avalanche Product Review Sentiment Dashboard")
st.caption("Explore customer reviews, sentiment scores, and product-level trends.")

# ---------------------------------------------------------------------------
# LOAD + VALIDATE + CLEAN
# ---------------------------------------------------------------------------
st.sidebar.header("Data")
raw_df, source = get_raw_dataframe()

if raw_df is None:
    st.info(
        "👋 No dataset loaded yet. Upload a CSV using the sidebar, or place a "
        f"file named **{DEFAULT_CSV_PATH}** in the app folder.\n\n"
        f"Required columns: {', '.join(REQUIRED_COLUMNS)}"
    )
    st.stop()

missing_cols = validate_columns(raw_df)
if missing_cols:
    st.error(
        "The uploaded dataset is missing the following required columns: "
        f"{', '.join(missing_cols)}"
    )
    st.stop()

if raw_df.empty:
    st.warning("The loaded dataset is empty. Please upload a file with review data.")
    st.stop()

df, n_dropped = clean_data(raw_df)

if df.empty:
    st.error(
        "None of the rows in this dataset could be used — check that "
        "SENTIMENT_SCORE contains numeric values and PRODUCT/SUMMARY are filled in."
    )
    st.stop()

st.success(f"Loaded {len(df)} valid reviews from {source}. ({n_dropped} row(s) skipped due to missing/invalid data.)")

# ---------------------------------------------------------------------------
# 9. SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

all_products = sorted(df["PRODUCT"].unique().tolist())
selected_products = st.sidebar.multiselect("Product", options=all_products, default=all_products)

sentiment_options = ["All", "Positive", "Neutral", "Negative"]
selected_sentiment = st.sidebar.selectbox("Sentiment category", sentiment_options)

# Date filter — only offer it if we have at least one valid date
valid_dates = df["DATE"].dropna()
if not valid_dates.empty:
    min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
else:
    date_range = None
    st.sidebar.caption("No valid dates found — date filter disabled.")

# Apply filters
filtered_df = df[df["PRODUCT"].isin(selected_products)]

if selected_sentiment != "All":
    filtered_df = filtered_df[filtered_df["SENTIMENT_CATEGORY"] == selected_sentiment]

if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    # Rows with missing dates are kept out of date-filtered results, but kept
    # in view when the full range is selected (i.e. no filtering happened).
    if not (start_date == min_date and end_date == max_date):
        filtered_df = filtered_df[
            (filtered_df["DATE"].dt.date >= start_date) & (filtered_df["DATE"].dt.date <= end_date)
        ]

# ---------------------------------------------------------------------------
# 6. KPI SECTION
# ---------------------------------------------------------------------------
st.subheader("Overview")

if filtered_df.empty:
    st.warning("No reviews match the current filters. Try widening your selection.")
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", len(filtered_df))
    col2.metric("Average Sentiment", f"{filtered_df['SENTIMENT_SCORE'].mean():.3f}")
    col3.metric("Positive Reviews", int((filtered_df["SENTIMENT_CATEGORY"] == "Positive").sum()))
    col4.metric("Negative Reviews", int((filtered_df["SENTIMENT_CATEGORY"] == "Negative").sum()))

    st.caption(
        f"Categories are based on SENTIMENT_SCORE thresholds: "
        f"Positive ≥ {POSITIVE_THRESHOLD}, Negative ≤ {NEGATIVE_THRESHOLD}, otherwise Neutral. "
        "Adjust these in the code if your dataset uses a different scale."
    )

# ---------------------------------------------------------------------------
# 7 & 8. SENTIMENT STATS + CHARTS
# ---------------------------------------------------------------------------
if not filtered_df.empty:
    st.subheader("Sentiment Analysis")

    stat_col1, stat_col2 = st.columns(2)

    with stat_col1:
        stats = {
            "Mean": filtered_df["SENTIMENT_SCORE"].mean(),
            "Median": filtered_df["SENTIMENT_SCORE"].median(),
            "Min": filtered_df["SENTIMENT_SCORE"].min(),
            "Max": filtered_df["SENTIMENT_SCORE"].max(),
        }
        st.write("**Summary statistics**")
        st.table(pd.Series(stats, name="Value").round(3))

    with stat_col2:
        fig_dist = px.histogram(
            filtered_df,
            x="SENTIMENT_SCORE",
            nbins=20,
            title="Sentiment Score Distribution",
            labels={"SENTIMENT_SCORE": "Sentiment Score", "count": "Number of Reviews"},
        )
        fig_dist.update_layout(yaxis_title="Number of Reviews")
        st.plotly_chart(fig_dist, use_container_width=True)

    # Product-level sentiment chart
    product_avg = (
        filtered_df.groupby("PRODUCT")["SENTIMENT_SCORE"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig_product = px.bar(
        product_avg,
        x="PRODUCT",
        y="SENTIMENT_SCORE",
        title="Average Sentiment by Product",
        labels={"SENTIMENT_SCORE": "Average Sentiment Score", "PRODUCT": "Product"},
    )
    st.plotly_chart(fig_product, use_container_width=True)

# ---------------------------------------------------------------------------
# 10. REVIEW TABLE
# ---------------------------------------------------------------------------
st.subheader("Reviews")
st.dataframe(
    filtered_df[["PRODUCT", "DATE", "SUMMARY", "SENTIMENT_SCORE", "SENTIMENT_CATEGORY"]],
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# 14. DOWNLOAD FEATURE
# ---------------------------------------------------------------------------
csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered data as CSV",
    data=csv_bytes,
    file_name="filtered_reviews.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# 11/12. NATURAL-LANGUAGE QUESTION INTERFACE
# ---------------------------------------------------------------------------
st.subheader("Ask Questions About the Reviews")
st.caption(
    "Try: \"What is the average sentiment?\", \"Which product has the highest average sentiment?\", "
    "\"How many reviews are available?\", \"Show me the most positive reviews.\""
)

question = st.text_input("Ask a question about the dataset...")

if question:
    answer = try_answer_with_pandas(question, filtered_df)
    if answer is None:
        with st.spinner("Thinking..."):
            answer = ask_llm(question, filtered_df)
    st.markdown(answer)
