# E-Library Recommendation System — Streamlit Edition

A Python web application that groups readers using K-Means clustering and recommends books based on positive ratings from readers in the same cluster. It is designed to run as one Streamlit application, avoiding the previous Vercel serverless upload-body limit that rejected large CSV requests with HTTP 413.

## Features
- Upload a book catalogue CSV and reader-ratings CSV.
- Column-name normalization for common Goodbooks-10k variants.
- Reader-book sparse matrix; TruncatedSVD feature reduction; K-Means or MiniBatchKMeans selected automatically by dataset size.
- Cluster summaries and personalized recommendations based on real uploaded data.
- Download recommendations and a JSON analysis summary.
- Optional MongoDB Atlas persistence for small analysis summaries; the app still works without MongoDB.
- Upload limit configured to 200 MB per file; actual hosting memory/CPU quotas still apply.
- Small sample CSV files included for a quick smoke test.

## Run locally (Windows / PowerShell)

```powershell
cd path\to\elibrary-streamlit
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local address printed by Streamlit (usually `http://localhost:8501`). Upload `data/books_sample.csv` and `data/ratings_sample.csv` to test the flow. The tiny sample is only a smoke test; use your real dataset for meaningful recommendations.

## Goodbooks-10k dataset
- `books.csv`: must include a book ID column (`book_id` or `id`) and preferably `title` / `original_title`.
- Ratings CSV: must include `user_id`, `book_id`, and `rating`. Goodbooks `ratings.csv` uses these columns. Ensure IDs in the two files refer to the same book-ID system; `ratings.csv` from Goodbooks generally matches `books.csv` by `book_id`.
- Extra columns are ignored. Invalid rows are discarded, ratings are restricted to 0–5, and duplicate user/book rows are averaged.
- For predictable resource use, the app keeps the most-rated books and most-active readers up to the selected limits. This is a practical sample of the dataset, not a claim that every row was used.

## Deploy on Streamlit Community Cloud
1. Create a new GitHub repository and upload all files in this ZIP.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Select **Create app**, choose the repository, branch `main`, and main file path `streamlit_app.py`.
4. Deploy. Do not deploy this Streamlit version to Vercel; Vercel's serverless request-body limit was the cause of the earlier HTTP 413 issue.
5. If you want MongoDB persistence, open app settings → **Secrets** and add:

```toml
MONGODB_URI = "mongodb+srv://YOUR_USER:YOUR_PASSWORD@YOUR_CLUSTER/..."
MONGODB_DB_NAME = "elibrary"
```

If these secrets are absent, the app runs without MongoDB. Keep database credentials out of GitHub. In MongoDB Atlas, allow network access from the deployment environment according to your security requirements; avoid leaving a broad IP allowlist in production if you can use a tighter option.

## Important deployment and data notes
- Streamlit Community Cloud has resource quotas; large data can still exceed available memory or CPU. Start with the default 5,000-reader / 300-book limits, then increase only after testing.
- `.streamlit/config.toml` sets the per-file upload limit to 200 MB. This is not a guarantee that a hosting provider can process a 200 MB dataset.
- Streamlit session state and uploaded files are temporary. The app does not upload raw CSVs to MongoDB. Only a compact summary is persisted when MongoDB is configured.
- The model and recommendation state live in the active session. Refreshing or restarting the app may require uploading and analyzing again.
- Recommendations are generated from ratings of readers in the same learned cluster; there are no hardcoded recommendation results.

## Project layout

```text
elibrary-streamlit/
├── streamlit_app.py
├── app/
│   ├── __init__.py
│   ├── recommender.py
│   └── database.py
├── data/
│   ├── books_sample.csv
│   └── ratings_sample.csv
├── .streamlit/config.toml
├── .env.example
├── .gitignore
└── requirements.txt
```
