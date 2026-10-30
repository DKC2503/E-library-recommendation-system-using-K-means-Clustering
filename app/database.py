from __future__ import annotations
from datetime import datetime, timezone
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError

class MongoStore:
    """Optional MongoDB Atlas persistence for small analysis summaries only."""
    def __init__(self, uri: str = "", db_name: str = "elibrary"):
        self.client = None
        self.collection = None
        self.error = None
        if uri:
            try:
                self.client = MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)
                self.client.admin.command("ping")
                self.collection = self.client[db_name]["analysis_history"]
            except Exception as exc:
                self.error = str(exc)
                self.client = None
                self.collection = None

    @property
    def available(self) -> bool:
        return self.collection is not None

    def save_analysis(self, summary: dict) -> None:
        if not self.available: return
        doc = dict(summary)
        doc["saved_at"] = datetime.now(timezone.utc).isoformat()
        self.collection.insert_one(doc)

    def get_history(self, limit: int = 10) -> list[dict]:
        if not self.available: return []
        docs = list(self.collection.find({}, {"_id": 0}).sort("saved_at", DESCENDING).limit(limit))
        return docs

    def close(self):
        if self.client:
            self.client.close()
