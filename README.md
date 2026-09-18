# Avalanche Sentiment Analysis (MVP)

A simple Streamlit dashboard for exploring Avalanche product reviews and understanding customer sentiment — no manual CSV-reading required.

## Project Description

This app loads a CSV of product reviews, cleans and validates the data, and gives you:

- An overview of average sentiment and review counts
- Charts showing how sentiment is distributed and how it varies by product
- Filters for product, sentiment category, and date
- A searchable/downloadable review table
- A simple question box for asking things like "What is the average sentiment?" — answered directly with Pandas, with an optional LLM fallback for more open-ended questions

It's intentionally minimal: no login, no database, no complex backend. Just upload a CSV and explore.

## Features

**Must-have (implemented):**
- CSV upload with validation and error handling
- Data cleaning (dates, numeric sentiment scores, missing values)
- Review table
- Average sentiment score and other KPIs
- Sentiment distribution chart
- Product-level sentiment summary chart
- Product and sentiment-category filters
- Natural-language question box (Pandas-first)

**Nice-to-have (implemented):**
- Date range filter
- Download filtered data as CSV
- Positive / Neutral / Negative sentiment categories (with adjustable thresholds)

**Not implemented (out of scope for this MVP):** authentication, databases, advanced RAG, multi-user accounts, real-time ingestion, enterprise deployment.

## Installation

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Enable the LLM question-answering fallback:

   ```bash
   cp .env.example .env
   # then edit .env and add your OPENAI_API_KEY
   ```

   The app works fully without this step — most questions are answered directly with Pandas, no API key needed.

## Running the App

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`) in your browser.

## Dataset Format

Your CSV needs these four columns:

| Column            | Description                                  |
|-------------------|-----------------------------------------------|
| `PRODUCT`         | Name of the product being reviewed             |
| `DATE`            | Date of the review (any common date format)    |
| `SUMMARY`         | The review text / summary                      |
| `SENTIMENT_SCORE` | A numeric sentiment score                       |

The app assumes `SENTIMENT_SCORE` is roughly on a **-1 (very negative) to 1 (very positive)** scale. If your data uses a different scale, open `app.py` and adjust the two constants near the top:

```python
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05
```

Rows with unusable sentiment scores, or missing `PRODUCT`/`SUMMARY`, are dropped automatically, and you'll see a message showing how many rows were skipped.

You can either:
- Upload a CSV through the sidebar, or
- Place a file named `avalanche_reviews.csv` in the same folder as `app.py` — the app will load it automatically if nothing is uploaded.

A small sample file (`avalanche_reviews.csv`) is included so you can try the app immediately.

## Environment Variables

Only needed if you want the optional LLM fallback for questions that Pandas can't answer directly (e.g. free-form or open-ended questions). This app uses the **Hugging Face Inference API** (free tier available).

1. Get a free access token at https://huggingface.co/settings/tokens (a "Read" token is enough).
2. Copy `.env.example` to `.env`.
3. Set `HF_TOKEN=` to your actual token.
4. Never commit `.env` or put a real token in this README — `.gitignore` already excludes `.env`.

**Deploying on Streamlit Community Cloud?** Add the token in your app's Settings → Secrets instead, using TOML format:
```toml
HF_TOKEN = "hf_your_actual_token_here"
```
The app checks `st.secrets` first, then falls back to `.env` for local development, so the same code works in both places.

## How It Works (short version)

- **Loading**: `st.file_uploader` lets you upload a CSV; if none is provided, the app looks for `avalanche_reviews.csv` locally.
- **Validation**: checks the four required columns exist before doing anything else.
- **Cleaning**: converts `DATE` and `SENTIMENT_SCORE` to proper types, drops unusable rows, and labels each review Positive/Neutral/Negative.
- **KPIs & charts**: computed with Pandas and drawn with Plotly (histogram for distribution, bar chart for per-product averages).
- **Filters**: sidebar controls narrow the dataframe; every metric, chart, and table below reacts to the filtered data.
- **Question box**: first tries to answer with plain Pandas (`try_answer_with_pandas`). If it doesn't recognize the question and an `HF_TOKEN` is set, it sends a small *summary* of the data (not the raw reviews) to a Hugging Face model with a strict system prompt telling it not to invent numbers.

## Test Cases to Run

1. **No file present** — launch the app with no upload and no local CSV → should show a friendly info message, not crash.
2. **Sample data** — use the included `avalanche_reviews.csv` → table, KPIs, and charts should populate.
3. **Missing column** — upload a CSV missing `SENTIMENT_SCORE` → should show a clear error and stop.
4. **Invalid sentiment values** — include a few rows with non-numeric `SENTIMENT_SCORE` (e.g. `"n/a"`) → those rows should be dropped, with a message showing the count skipped.
5. **Invalid/missing dates** — include a malformed date → row is kept (unless sentiment is also invalid), date filter still works for the rest.
6. **Empty dataset** — upload a CSV with headers only → should show a warning, not crash.
7. **Filters producing zero rows** — pick a product + sentiment combo with no matches → should show a "no reviews match" warning instead of breaking charts.
8. **Download button** — filter the data, click download, confirm the CSV matches what's on screen.
9. **Pandas questions** — try each of the example questions and confirm sensible, correctly calculated answers.
10. **No API token** — leave `.env` unset, ask an open-ended question the Pandas logic doesn't recognize → should explain that no Hugging Face token is configured, not crash.
11. **With API token** — set a valid `HF_TOKEN`, ask an open-ended question → should get a relevant answer grounded in the data summary.
