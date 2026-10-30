# E-Library K-Means Recommendation System (Python)

Python-first mini project: FastAPI backend, browser frontend, scikit-learn K-Means clustering, personalized recommendations, optional MongoDB Atlas persistence.

## Run on Windows
Use Python 3.12 if possible (scientific libraries may not support every newest Python version).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and paste your Atlas URI after MONGODB_URI=
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000.

Upload `books.csv` (requires `book_id,title`) and `ratings.csv` (requires `user_id,book_id,rating`). Default maximum is 100 MiB per file.

## Algorithm
1. Validate and clean CSV data.
2. Select the most-rated books above the chosen threshold.
3. Construct a sparse reader-book matrix.
4. Use TruncatedSVD to reduce feature dimensions when possible.
5. Fit real scikit-learn MiniBatchKMeans (or KMeans for small data).
6. Summarize clusters and recommend unrated books using a damped cluster rating score.
7. Unknown reader IDs receive community-wide cold-start recommendations.

## MongoDB
Set `MONGODB_URI` in `.env` to enable persistence. The app stores analysis history, dataset summaries, and cluster summaries. It continues working without MongoDB. Never commit `.env` or expose your URI. The full ratings CSV is not stored as millions of MongoDB documents.

## API
`GET /api/health`, `POST /api/analyze`, `GET /api/clusters`, `GET /api/recommend/{reader_id}`, `GET /api/history`, `GET /api/dataset/status`. Interactive API docs: `/docs`.

## Tests
`pytest -q`

Current model state is in memory; after restarting the app, upload and analyze again. MongoDB preserves history and summary records, not a serialized full model/matrix. Keep original datasets backed up.
