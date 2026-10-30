from __future__ import annotations
import os
import time
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from app.recommender import analyze_dataset, recommend_for_reader, normalize_books, normalize_ratings
from app.database import MongoStore

st.set_page_config(page_title="E-Library | Smart Recommendations", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { color-scheme: light; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1450px; }
.hero { padding: 1.6rem 1.8rem; border-radius: 20px; background: linear-gradient(120deg,#f7eadb,#fdf8f1); border: 1px solid #ead7c3; margin-bottom: 1.2rem; }
.hero h1 { margin: 0; color: #34261d; font-size: 2.25rem; }
.hero p { margin: .45rem 0 0; color: #725c4c; font-size: 1.03rem; }
div[data-testid="stMetric"] { background: #fffaf4; border: 1px solid #eadfd3; padding: 1rem; border-radius: 14px; }
section[data-testid="stSidebar"] { background: #fbf4eb; }
.small-note { color: #75685e; font-size: .9rem; }
</style>
<div class="hero"><h1>📚 E-Library Recommendation System</h1><p>Reader clustering with K-Means · Personalized book suggestions · Real dataset analysis</p></div>
""", unsafe_allow_html=True)

@st.cache_resource
def mongo_store():
    # Secrets are optional; a missing secrets.toml must not prevent local startup.
    try:
        uri = st.secrets.get("MONGODB_URI", os.getenv("MONGODB_URI", ""))
        db_name = st.secrets.get("MONGODB_DB_NAME", os.getenv("MONGODB_DB_NAME", "elibrary"))
    except Exception:
        uri = os.getenv("MONGODB_URI", "")
        db_name = os.getenv("MONGODB_DB_NAME", "elibrary")
    return MongoStore(uri, db_name)

with st.sidebar:
    st.header("⚙️ Analysis settings")
    n_clusters = st.slider("Reader clusters (K)", 2, 12, 5)
    threshold = st.slider("Positive rating threshold", 1.0, 5.0, 3.0, 0.5)
    max_books = st.select_slider("Maximum book features", options=[100, 200, 300, 500, 750], value=300)
    max_readers = st.select_slider("Maximum readers sampled", options=[1000, 2500, 5000, 7500, 10000], value=5000)
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
    st.divider()
    st.caption("Designed for Goodbooks-10k-style CSVs. Limits keep memory use predictable on free hosting.")

with st.expander("📄 Expected CSV format", expanded=False):
    st.markdown("**books.csv** must contain `book_id` and a title column such as `title` or `original_title`. **ratings.csv** must contain `user_id`, `book_id`, and `rating`. Extra columns are allowed. The app also recognizes common variants like `userId`, `bookId`, and `ratings`.")
    st.code("# books.csv\nbook_id,title,authors\n1,The Example Book,Example Author\n\n# ratings.csv\nuser_id,book_id,rating\n10,1,5\n10,2,4\n11,1,3", language="csv")

st.subheader("1. Upload your data")
col1, col2 = st.columns(2)
with col1:
    books_file = st.file_uploader("Books catalogue (CSV)", type=["csv"], key="books", help="Goodbooks-10k books.csv is supported.")
with col2:
    ratings_file = st.file_uploader("Reader ratings (CSV)", type=["csv"], key="ratings", help="Goodbooks-10k ratings.csv is supported.")

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🚀 Analyze dataset and build recommendations", type="primary", use_container_width=True, disabled=not (books_file and ratings_file)):
    try:
        with st.spinner("Reading CSV files and preparing reader-book features…"):
            books_raw = pd.read_csv(books_file, low_memory=False)
            ratings_raw = pd.read_csv(ratings_file, low_memory=False)
            books = normalize_books(books_raw)
            ratings = normalize_ratings(ratings_raw)
        if books.empty or ratings.empty:
            st.error("The uploaded files contain no usable rows after validation.")
        else:
            progress = st.progress(0, text="Preparing data…")
            result = analyze_dataset(books, ratings, n_clusters=int(n_clusters), rating_threshold=float(threshold), max_books=int(max_books), max_readers=int(max_readers), random_state=int(seed), progress_callback=lambda p, msg: progress.progress(p, text=msg))
            result["created_at"] = datetime.now(timezone.utc).isoformat()
            st.session_state.analysis = result
            st.session_state.history.insert(0, {"created_at": result["created_at"], "readers": result["n_readers"], "books": result["n_books"], "ratings": result["n_ratings"], "clusters": result["n_clusters"], "algorithm": result["algorithm"]})
            st.session_state.history = st.session_state.history[:10]
            store = mongo_store()
            if store.available:
                try:
                    store.save_analysis({
                        "run_id": result["run_id"], "created_at": result["created_at"],
                        "algorithm": result["algorithm"], "random_state": result["random_state"],
                        "n_clusters": result["n_clusters"], "n_readers": result["n_readers"],
                        "n_books": result["n_books"], "n_ratings": result["n_ratings"],
                        "rating_threshold": result["rating_threshold"], "max_books": result["max_books"],
                        "runtime_seconds": result["runtime_seconds"],
                        "cluster_counts_dict": result["cluster_counts_dict"],
                    })
                    st.toast("Analysis summary saved to MongoDB Atlas.")
                except Exception as e:
                    st.warning(f"Analysis completed, but MongoDB save failed: {e}")
            progress.progress(100, text="Analysis complete")
            st.success("Analysis completed successfully. Your recommendations are generated from the uploaded ratings.")
    except Exception as e:
        st.error(f"Could not analyze these files: {e}")
        st.info("Check that the CSVs contain book_id/title and user_id/book_id/rating columns, and that the rating values are numeric.")

result = st.session_state.analysis
if result:
    st.divider()
    st.subheader("2. Dataset overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Readers analyzed", f"{result['n_readers']:,}")
    m2.metric("Books analyzed", f"{result['n_books']:,}")
    m3.metric("Ratings used", f"{result['n_ratings']:,}")
    m4.metric("Reader clusters", result["n_clusters"])
    m5.metric("Positive rating threshold", f"{result['rating_threshold']:.1f} / 5")
    st.caption(f"Algorithm: {result['algorithm']} · Seed: {result['random_state']} · Runtime: {result['runtime_seconds']:.2f}s · Run: {result['run_id']}")

    st.subheader("3. Explore reader clusters")
    cluster_counts = result["cluster_counts"].rename_axis("Cluster").reset_index(name="Readers")
    a, b = st.columns([1, 1.4])
    with a:
        st.bar_chart(cluster_counts.set_index("Cluster"))
        st.dataframe(cluster_counts, use_container_width=True, hide_index=True)
    with b:
        selected_cluster = st.selectbox("Inspect cluster", options=sorted(result["cluster_counts"].index.tolist()), format_func=lambda x: f"Cluster {x} ({result['cluster_counts'].loc[x]:,} readers)")
        top_cluster_books = result["cluster_top_books"].get(int(selected_cluster), pd.DataFrame())
        st.markdown("**Most positively rated books in this cluster**")
        if not top_cluster_books.empty:
            st.dataframe(top_cluster_books, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough positive ratings to summarize this cluster.")

    st.subheader("4. Personalized recommendations")
    reader_ids = result["reader_ids"]
    chosen_reader = st.selectbox("Choose a reader ID", options=reader_ids, index=0 if reader_ids else None, format_func=str)
    count = st.slider("Number of recommendations", 5, 20, 10, key="recommend_count")
    if chosen_reader is not None:
        recs = recommend_for_reader(result, chosen_reader, limit=count)
        if recs.empty:
            st.warning("No unseen recommendations are available for this reader. Try another reader or lower the positive threshold.")
        else:
            st.dataframe(recs, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download recommendations CSV", recs.to_csv(index=False).encode("utf-8-sig"), file_name=f"recommendations_reader_{chosen_reader}.csv", mime="text/csv")

    st.subheader("5. Export analysis")
    summary = {k: v for k, v in result.items() if k in ["run_id", "created_at", "algorithm", "random_state", "n_clusters", "n_readers", "n_books", "n_ratings", "rating_threshold", "max_books", "runtime_seconds", "cluster_counts_dict"]}
    st.download_button("⬇️ Download analysis summary (JSON)", data=__import__("json").dumps(summary, indent=2, default=str), file_name="elibrary_analysis_summary.json", mime="application/json")

with st.expander("🕘 Recent analyses", expanded=False):
    store = mongo_store()
    db_history = []
    if store.available:
        try:
            db_history = store.get_history(limit=10)
        except Exception:
            pass
    combined = db_history or st.session_state.history
    if combined:
        st.dataframe(pd.DataFrame(combined), use_container_width=True, hide_index=True)
    else:
        st.caption("No analyses yet. History is stored in this session and optionally in MongoDB Atlas when configured.")

st.divider()
st.markdown('<p class="small-note">Privacy note: uploaded CSVs are processed for the current analysis. MongoDB stores analysis summaries only, not the raw uploaded files. On hosted free instances, session data may reset when the app restarts.</p>', unsafe_allow_html=True)
