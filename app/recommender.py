from __future__ import annotations
import time, uuid, re
from typing import Callable, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lookup = {_key(c): c for c in df.columns}
    for candidate in candidates:
        if _key(candidate) in lookup:
            return lookup[_key(candidate)]
    return None


def normalize_books(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Books CSV is empty.")
    id_col = _find_column(df, ["book_id", "bookId", "id", "goodreads_book_id"])
    title_col = _find_column(df, ["title", "original_title", "book_title", "name"])
    if not id_col:
        raise ValueError("Books CSV needs a book_id (or id) column.")
    out = df.copy()
    out["book_id"] = pd.to_numeric(out[id_col], errors="coerce")
    if title_col:
        out["title"] = out[title_col].fillna("").astype(str).str.strip()
    else:
        out["title"] = "Book " + out["book_id"].astype(str)
    author_col = _find_column(out, ["authors", "author", "book_author"])
    out["authors"] = out[author_col].fillna("").astype(str) if author_col else ""
    out = out.dropna(subset=["book_id"]).drop_duplicates("book_id", keep="first")
    out["book_id"] = out["book_id"].astype(int)
    out["title"] = out["title"].replace("", np.nan).fillna(out["book_id"].map(lambda x: f"Book {x}"))
    return out[["book_id", "title", "authors"]].reset_index(drop=True)


def normalize_ratings(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Ratings CSV is empty.")
    user_col = _find_column(df, ["user_id", "userId", "reader_id", "readerId", "userid"])
    book_col = _find_column(df, ["book_id", "bookId", "goodreads_book_id", "item_id"])
    rating_col = _find_column(df, ["rating", "ratings", "score", "book_rating"])
    if not user_col or not book_col or not rating_col:
        raise ValueError("Ratings CSV needs user_id, book_id, and rating columns.")
    out = pd.DataFrame({"user_id": pd.to_numeric(df[user_col], errors="coerce"), "book_id": pd.to_numeric(df[book_col], errors="coerce"), "rating": pd.to_numeric(df[rating_col], errors="coerce")})
    out = out.dropna(subset=["user_id", "book_id", "rating"])
    out = out[(out["rating"] >= 0) & (out["rating"] <= 5)]
    out["user_id"] = out["user_id"].astype(int)
    out["book_id"] = out["book_id"].astype(int)
    # Multiple ratings for the same user/book are averaged rather than double-counted.
    return out.groupby(["user_id", "book_id"], as_index=False, sort=False)["rating"].mean()


def analyze_dataset(books: pd.DataFrame, ratings: pd.DataFrame, *, n_clusters: int = 5, rating_threshold: float = 3.0, max_books: int = 300, max_readers: int = 5000, random_state: int = 42, progress_callback: Optional[Callable[[int, str], None]] = None) -> dict:
    start = time.perf_counter()
    def progress(p, msg):
        if progress_callback: progress_callback(int(p), msg)
    if len(books) == 0 or len(ratings) == 0:
        raise ValueError("No usable books or ratings were found.")
    if n_clusters < 2:
        raise ValueError("K must be at least 2.")
    progress(5, "Matching ratings to known books…")
    book_ids = set(books["book_id"].tolist())
    ratings = ratings[ratings["book_id"].isin(book_ids)].copy()
    if ratings.empty:
        raise ValueError("No ratings match the book IDs in books.csv. Confirm both files use the same ID system.")
    # Keep the most-rated books and most-active readers to bound memory on hosted deployments.
    book_counts = ratings.groupby("book_id").size().sort_values(ascending=False)
    selected_book_ids = book_counts.head(max_books).index
    ratings = ratings[ratings["book_id"].isin(selected_book_ids)]
    user_counts = ratings.groupby("user_id").size().sort_values(ascending=False)
    selected_user_ids = user_counts.head(max_readers).index
    ratings = ratings[ratings["user_id"].isin(selected_user_ids)].copy()
    if ratings.empty:
        raise ValueError("No ratings remain after applying the dataset limits.")
    user_ids = sorted(ratings["user_id"].unique().tolist())
    used_book_ids = sorted(ratings["book_id"].unique().tolist())
    actual_k = min(int(n_clusters), len(user_ids))
    if actual_k < 2:
        raise ValueError("At least two distinct readers are required for K-Means.")
    progress(20, f"Building sparse reader-book matrix ({len(user_ids):,} readers × {len(used_book_ids):,} books)…")
    user_index = {u: i for i, u in enumerate(user_ids)}
    book_index = {b: i for i, b in enumerate(used_book_ids)}
    row = ratings["user_id"].map(user_index).to_numpy(dtype=np.int32)
    col = ratings["book_id"].map(book_index).to_numpy(dtype=np.int32)
    vals = ratings["rating"].to_numpy(dtype=np.float32)
    matrix = csr_matrix((vals, (row, col)), shape=(len(user_ids), len(used_book_ids)), dtype=np.float32)
    progress(35, "Reducing sparse features for clustering…")
    # SVD helps reduce sparsity/noise while keeping memory use bounded.
    n_components = min(30, max(2, min(matrix.shape) - 1))
    if min(matrix.shape) >= 3:
        svd = TruncatedSVD(n_components=n_components, random_state=random_state, n_iter=5)
        vectors = svd.fit_transform(matrix)
        vectors = normalize(vectors, norm="l2", copy=False)
    else:
        svd = None
        vectors = matrix.toarray()
    progress(65, "Clustering reader profiles with MiniBatchKMeans…")
    if len(user_ids) >= 1000:
        model = MiniBatchKMeans(n_clusters=actual_k, random_state=random_state, batch_size=min(1024, max(128, len(user_ids)//10)), n_init=3, max_iter=100)
    else:
        model = KMeans(n_clusters=actual_k, random_state=random_state, n_init=10, max_iter=200)
    labels = model.fit_predict(vectors)
    progress(80, "Summarizing clusters and popular books…")
    book_meta = books.set_index("book_id")
    positive = ratings[ratings["rating"] >= rating_threshold]
    cluster_counts = pd.Series(labels, index=user_ids, name="cluster").value_counts().sort_index()
    cluster_top_books = {}
    for cluster in sorted(set(labels.tolist())):
        members = set(np.asarray(user_ids)[labels == cluster].tolist())
        subset = positive[positive["user_id"].isin(members)]
        if subset.empty: cluster_top_books[int(cluster)] = pd.DataFrame(columns=["book_id", "title", "authors", "average_rating", "positive_ratings"]); continue
        stats = subset.groupby("book_id").agg(average_rating=("rating", "mean"), positive_ratings=("rating", "size")).reset_index()
        stats = stats.sort_values(["positive_ratings", "average_rating"], ascending=False).head(10)
        stats["title"] = stats["book_id"].map(book_meta["title"]).fillna("Unknown title")
        stats["authors"] = stats["book_id"].map(book_meta["authors"]).fillna("")
        cluster_top_books[int(cluster)] = stats[["book_id", "title", "authors", "average_rating", "positive_ratings"]].reset_index(drop=True)
    progress(95, "Finalizing analysis…")
    result = {
        "run_id": str(uuid.uuid4())[:8], "algorithm": "MiniBatchKMeans" if isinstance(model, MiniBatchKMeans) else "KMeans",
        "random_state": int(random_state), "n_clusters": int(actual_k), "rating_threshold": float(rating_threshold),
        "max_books": int(max_books), "max_readers": int(max_readers), "n_readers": int(len(user_ids)),
        "n_books": int(len(used_book_ids)), "n_ratings": int(len(ratings)), "runtime_seconds": time.perf_counter()-start,
        "reader_ids": user_ids, "labels": labels, "reader_cluster": dict(zip(user_ids, labels.astype(int).tolist())),
        "cluster_counts": cluster_counts, "cluster_counts_dict": {int(k): int(v) for k,v in cluster_counts.items()},
        "cluster_top_books": cluster_top_books, "books_df": books, "ratings_df": ratings,
        "matrix": matrix, "model": model, "svd_model": svd, "reader_vectors": vectors,
        "seen_by_reader": ratings.groupby("user_id")["book_id"].apply(set).to_dict(),
        "book_meta": book_meta, "positive_ratings": positive,
    }
    progress(100, "Done")
    return result


def recommend_for_reader(result: dict, reader_id, limit: int = 10) -> pd.DataFrame:
    try: reader_id = int(reader_id)
    except (ValueError, TypeError): return pd.DataFrame()
    cluster = result["reader_cluster"].get(reader_id)
    if cluster is None: return pd.DataFrame()
    ratings = result["ratings_df"]
    seen = result["seen_by_reader"].get(reader_id, set())
    members = set(u for u,c in result["reader_cluster"].items() if c == cluster)
    peers = ratings[(ratings["user_id"].isin(members)) & (ratings["rating"] >= result["rating_threshold"]) & (~ratings["book_id"].isin(seen))]
    if peers.empty: return pd.DataFrame(columns=["book_id", "title", "authors", "average_rating", "positive_ratings", "cluster"])
    stats = peers.groupby("book_id").agg(average_rating=("rating", "mean"), positive_ratings=("rating", "size"), peer_readers=("user_id", "nunique")).reset_index()
    # Popularity + average rating, with a mild confidence boost for books liked by many cluster members.
    stats["score"] = stats["average_rating"] * 0.7 + (stats["positive_ratings"] / max(1, stats["positive_ratings"].max())) * 1.5
    meta = result["book_meta"]
    stats["title"] = stats["book_id"].map(meta["title"]).fillna("Unknown title")
    stats["authors"] = stats["book_id"].map(meta["authors"]).fillna("")
    stats["cluster"] = int(cluster)
    stats = stats.sort_values(["score", "peer_readers", "average_rating"], ascending=False).head(int(limit))
    stats["average_rating"] = stats["average_rating"].round(2)
    stats["score"] = stats["score"].round(2)
    return stats[["book_id", "title", "authors", "average_rating", "positive_ratings", "peer_readers", "score", "cluster"]].reset_index(drop=True)
