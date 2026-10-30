"""MongoDB persistence for analysis metadata and recommendation data.

The full sparse matrix is deliberately not stored in MongoDB. Compact per-reader
assignments/history and per-book aggregate statistics are stored instead.
"""
import logging
from pymongo import MongoClient, ASCENDING, DESCENDING, ReplaceOne
from app.config import MONGODB_URI, MONGODB_DB_NAME

log = logging.getLogger("elibrary")
client = None
db = None
last_error = None


def connect():
    global client, db, last_error
    if not MONGODB_URI:
        log.info("MONGODB_URI not set; persistence disabled.")
        return
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        client.admin.command("ping")
        db = client[MONGODB_DB_NAME]
        db.analysis_runs.create_index([("created_at", DESCENDING)])
        db.cluster_summaries.create_index([("run_id", ASCENDING), ("cluster_id", ASCENDING)], unique=True)
        db.dataset_catalog.create_index([("dataset_id", ASCENDING)], unique=True)
        db.reader_assignments.create_index([("run_id", ASCENDING), ("user_id", ASCENDING)], unique=True)
        db.reader_histories.create_index([("run_id", ASCENDING), ("user_id", ASCENDING)], unique=True)
        db.recommendation_stats.create_index([("run_id", ASCENDING), ("scope", ASCENDING), ("cluster_id", ASCENDING), ("book_id", ASCENDING)], unique=True)
        db.books.create_index([("book_id", ASCENDING)], unique=True)
        last_error = None
        log.info("MongoDB connected; indexes checked.")
    except Exception as exc:
        last_error = str(exc)
        log.warning("MongoDB unavailable; app will continue without persistence: %s", exc)
        if client:
            client.close()
        client, db = None, None


def health():
    if not MONGODB_URI:
        return {"configured": False, "connected": False, "message": "Set MONGODB_URI in .env to enable persistence."}
    if db is None:
        return {"configured": True, "connected": False, "message": last_error or "Not connected"}
    try:
        client.admin.command("ping")
        return {"configured": True, "connected": True, "database": MONGODB_DB_NAME}
    except Exception as exc:
        return {"configured": True, "connected": False, "message": str(exc)}


def get_db():
    return db


def persist_analysis(result, public):
    """Persist compact, queryable data needed to recover recommendations after restart."""
    if db is None:
        return False
    run_id = public["run_id"]
    # Catalogue metadata is upserted, not appended, so repeated runs do not duplicate books.
    book_ops = []
    for bid, meta in result["_book_meta"].items():
        doc = {"book_id": str(bid), "title": str(meta.get("title", "Untitled")),
               "authors": str(meta.get("authors", "Unknown author"))}
        book_ops.append(ReplaceOne({"book_id": str(bid)}, doc, upsert=True))
    if book_ops:
        for i in range(0, len(book_ops), 1000):
            db.books.bulk_write(book_ops[i:i+1000], ordered=False)

    assignment_ops = [ReplaceOne({"run_id": run_id, "user_id": str(uid)},
        {"run_id": run_id, "user_id": str(uid), "cluster_id": int(cid)}, upsert=True)
        for uid, cid in result["_user_cluster"].items()]
    for i in range(0, len(assignment_ops), 1000):
        db.reader_assignments.bulk_write(assignment_ops[i:i+1000], ordered=False)

    history_ops = [ReplaceOne({"run_id": run_id, "user_id": str(uid)},
        {"run_id": run_id, "user_id": str(uid), "book_ids": sorted(map(str, bids))}, upsert=True)
        for uid, bids in result["_rated"].items()]
    for i in range(0, len(history_ops), 500):
        db.reader_histories.bulk_write(history_ops[i:i+500], ordered=False)

    stat_ops = []
    for cid, stats in result["_cluster_stats"].items():
        for bid, st in stats.items():
            stat_ops.append(ReplaceOne({"run_id": run_id, "scope": "cluster", "cluster_id": int(cid), "book_id": str(bid)},
                {"run_id": run_id, "scope": "cluster", "cluster_id": int(cid), "book_id": str(bid),
                 "mean": float(st["mean"]), "count": int(st["count"])}, upsert=True))
    for bid, st in result["_global_stats"].items():
        stat_ops.append(ReplaceOne({"run_id": run_id, "scope": "global", "cluster_id": -1, "book_id": str(bid)},
            {"run_id": run_id, "scope": "global", "cluster_id": -1, "book_id": str(bid),
             "mean": float(st["mean"]), "count": int(st["count"])}, upsert=True))
    for i in range(0, len(stat_ops), 1000):
        db.recommendation_stats.bulk_write(stat_ops[i:i+1000], ordered=False)

    db.analysis_runs.replace_one({"run_id": run_id}, {**public, "global_mean": float(result["_global_mean"])}, upsert=True)
    db.dataset_catalog.update_one({"dataset_id": result["dataset_id"]},
        {"$set": {"dataset_id": result["dataset_id"], "latest_run_id": run_id,
                   "updated_at": public["created_at"], "metrics": result["metrics"],
                   "parameters": result["parameters"], "global_mean": float(result["_global_mean"])}}, upsert=True)
    db.cluster_summaries.delete_many({"run_id": run_id})
    if result["clusters"]:
        db.cluster_summaries.insert_many([{"run_id": run_id, **c} for c in result["clusters"]])
    return True


def latest_run():
    if db is None:
        return None
    return db.analysis_runs.find_one({}, {"_id": 0}, sort=[("created_at", DESCENDING)])


def load_recommendations(reader_id, limit=10):
    """Build recommendations directly from persisted aggregates without loading all users."""
    if db is None:
        return None
    run = latest_run()
    if not run:
        return None
    run_id = run.get("run_id")
    assignment = db.reader_assignments.find_one({"run_id": run_id, "user_id": str(reader_id)}, {"_id": 0, "cluster_id": 1})
    cold = assignment is None
    cid = None if cold else int(assignment["cluster_id"])
    query = {"run_id": run_id, "scope": "global", "cluster_id": -1} if cold else {
        "run_id": run_id, "scope": "cluster", "cluster_id": cid}
    seen_doc = db.reader_histories.find_one({"run_id": run_id, "user_id": str(reader_id)}, {"_id": 0, "book_ids": 1})
    seen = set(seen_doc.get("book_ids", [])) if seen_doc else set()
    global_mean = float(run.get("global_mean", 0))
    items = []
    stats = db.recommendation_stats.find(query, {"_id": 0}).sort([("mean", DESCENDING), ("count", DESCENDING)]).limit(5000)
    for st in stats:
        bid = str(st["book_id"])
        if bid in seen:
            continue
        meta = db.books.find_one({"book_id": bid}, {"_id": 0, "title": 1, "authors": 1})
        if not meta:
            continue
        n, mean = float(st["count"]), float(st["mean"])
        score = (n / (n + 5.0)) * mean + (5.0 / (n + 5.0)) * global_mean
        reason = "Community favourite (cold-start fallback)" if cold else f"Popular among readers in Cluster {cid}; based on {int(n)} ratings"
        items.append({"book_id": bid, "title": meta.get("title", "Untitled"),
                      "authors": meta.get("authors", "Unknown author"), "score": round(score, 3),
                      "average_rating": round(mean, 3), "rating_count": int(n), "reason": reason})
        if len(items) >= max(1, min(int(limit), 50)):
            break
    return {"reader_id": str(reader_id), "cluster_id": cid, "is_cold_start": cold,
            "run_id": run_id, "recommendations": items, "source": "MongoDB persisted analysis"}


def close():
    global client, db
    if client:
        client.close()
    client, db = None, None
