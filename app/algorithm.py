import hashlib, time
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

def run_kmeans(books_df, ratings_df, n_clusters=5, rating_threshold=3.0,
               algorithm="minibatch_kmeans", random_state=42,
               max_feature_books=500, max_readers=100000):
    started = time.perf_counter()
    missing = {"book_id", "title"} - set(books_df.columns)
    if missing:
        raise ValueError("books.csv missing columns: " + ", ".join(sorted(missing)))
    missing = {"user_id", "book_id", "rating"} - set(ratings_df.columns)
    if missing:
        raise ValueError("ratings.csv missing columns: " + ", ".join(sorted(missing)))
    books = books_df.copy()
    books["book_id"] = books["book_id"].astype(str).str.strip()
    books["title"] = books["title"].fillna("Untitled").astype(str)
    books = books[books.book_id.ne("")].drop_duplicates("book_id")
    ratings = ratings_df[["user_id", "book_id", "rating"]].copy()
    ratings["user_id"] = ratings.user_id.astype(str).str.strip()
    ratings["book_id"] = ratings.book_id.astype(str).str.strip()
    ratings["rating"] = pd.to_numeric(ratings.rating, errors="coerce")
    ratings = ratings.dropna(subset=["rating"])
    ratings = ratings[ratings.rating.between(0, 5) & ratings.user_id.ne("") & ratings.book_id.ne("")]
    ratings = ratings[ratings.book_id.isin(set(books.book_id))]
    ratings = ratings.groupby(["user_id", "book_id"], as_index=False, sort=False).rating.mean()
    if ratings.empty:
        raise ValueError("No valid ratings match book IDs in books.csv.")
    active = ratings[ratings.rating >= float(rating_threshold)]
    if active.empty:
        raise ValueError("No ratings meet the threshold. Lower the minimum rating.")
    feature_ids = active.book_id.value_counts().head(max_feature_books).index.astype(str).tolist()
    active = active[active.book_id.isin(feature_ids)]
    reader_ids = active.user_id.value_counts().head(max_readers).index.astype(str).tolist()
    active = active[active.user_id.isin(reader_ids)]
    reader_ids = sorted(active.user_id.unique().tolist())
    feature_ids = [b for b in feature_ids if b in set(active.book_id)]
    if len(reader_ids) < 2 or len(feature_ids) < 2:
        raise ValueError("At least two usable readers and two feature books are required.")
    ui = {u:i for i,u in enumerate(reader_ids)}
    bi = {b:i for i,b in enumerate(feature_ids)}
    row = active.user_id.map(ui).to_numpy(dtype=np.int32)
    col = active.book_id.map(bi).to_numpy(dtype=np.int32)
    vals = active.rating.to_numpy(dtype=np.float32) / 5.0
    matrix = csr_matrix((vals,(row,col)), shape=(len(reader_ids),len(feature_ids)), dtype=np.float32)
    matrix = normalize(matrix, norm="l2", axis=1, copy=False)
    dim = min(30, matrix.shape[0]-1, matrix.shape[1]-1)
    if dim >= 2:
        svd = TruncatedSVD(n_components=dim, random_state=int(random_state))
        projected = svd.fit_transform(matrix)
        variance = float(svd.explained_variance_ratio_.sum())
    else:
        projected, variance = matrix.toarray(), 0.0
    k = min(max(2,int(n_clusters)),len(reader_ids))
    if algorithm == "kmeans" and len(reader_ids) <= 5000:
        model = KMeans(n_clusters=k, random_state=int(random_state), n_init=3, max_iter=100)
        algo_used = "kmeans"
    else:
        model = MiniBatchKMeans(n_clusters=k, random_state=int(random_state), n_init=3,
                                max_iter=100, batch_size=min(1024,max(32,len(reader_ids))))
        algo_used = "minibatch_kmeans"
    labels = model.fit_predict(projected)
    user_cluster = {u:int(labels[i]) for i,u in enumerate(reader_ids)}
    all_ratings = ratings.copy()
    global_mean = float(all_ratings.rating.mean())
    book_meta = books.set_index("book_id",drop=False).to_dict("index")
    global_stats = all_ratings.groupby("book_id").rating.agg(["mean","count"]).to_dict("index")
    cluster_stats, clusters = {}, []
    for cid in range(k):
        members = {u for u,c in user_cluster.items() if c == cid}
        cr = all_ratings[all_ratings.user_id.isin(members)]
        stats = cr.groupby("book_id").rating.agg(["mean","count"]).to_dict("index")
        cluster_stats[cid] = stats
        top = []
        for bid, st in sorted(stats.items(), key=lambda x:(x[1]["mean"],x[1]["count"]), reverse=True):
            meta = book_meta.get(str(bid))
            if meta and len(top)<5:
                top.append({"book_id":str(bid),"title":str(meta["title"]),
                            "average_rating":round(float(st["mean"]),3),"rating_count":int(st["count"])})
        clusters.append({"cluster_id":cid,"reader_count":len(members),
                         "percentage":round(100*len(members)/len(reader_ids),2),
                         "average_rating":round(float(cr.rating.mean()),3) if len(cr) else 0,
                         "top_books":top})
    rated = defaultdict(set)
    for u,b in all_ratings[["user_id","book_id"]].itertuples(index=False,name=None):
        rated[str(u)].add(str(b))
    fingerprint = hashlib.sha256(f"{len(books)}:{len(all_ratings)}:{books.book_id.min()}:{books.book_id.max()}:{all_ratings.user_id.nunique()}".encode()).hexdigest()[:16]
    return {"dataset_id":fingerprint,"parameters":{"n_clusters":k,"rating_threshold":float(rating_threshold),
            "random_state":int(random_state),"algorithm":algo_used},
            "metrics":{"usable_readers":len(reader_ids),"usable_books":len(feature_ids),
            "rating_rows":len(all_ratings),"inertia":round(float(getattr(model,"inertia_",0)),4),
            "iterations":int(getattr(model,"n_iter_",0)),"explained_variance_ratio":round(variance,4),
            "duration_seconds":round(time.perf_counter()-started,3),"sparse_nonzeros":int(matrix.nnz)},
            "clusters":clusters,"_user_cluster":user_cluster,"_cluster_stats":cluster_stats,
            "_global_stats":global_stats,"_book_meta":book_meta,"_rated":dict(rated),
            "_global_mean":global_mean}

def recommend(result, reader_id, limit=10):
    uid = str(reader_id).strip()
    cold = uid not in result["_user_cluster"]
    cid = None if cold else result["_user_cluster"][uid]
    stats = result["_global_stats"] if cold else result["_cluster_stats"].get(cid,{})
    seen = result["_rated"].get(uid,set())
    global_mean = result["_global_mean"]
    reason = "Community favorites (cold-start fallback)" if cold else f"Popular among readers in Cluster {cid}"
    items=[]
    for bid,st in stats.items():
        bid=str(bid)
        if bid in seen: continue
        meta=result["_book_meta"].get(bid)
        if not meta: continue
        n=float(st["count"]); mean=float(st["mean"]); score=(n/(n+5))*mean+(5/(n+5))*global_mean
        items.append({"book_id":bid,"title":str(meta.get("title","Untitled")),
          "authors":str(meta.get("authors","Unknown author")),"score":round(score,3),
          "average_rating":round(mean,3),"rating_count":int(n),"reason":reason})
    items.sort(key=lambda x:(x["score"],x["rating_count"]),reverse=True)
    return {"reader_id":uid,"cluster_id":cid,"is_cold_start":cold,"recommendations":items[:max(1,min(int(limit),50))]}
