import os
from dotenv import load_dotenv
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "elibrary").strip() or "elibrary"
MAX_UPLOAD_MB = max(1, int(os.getenv("MAX_UPLOAD_MB", "100")))
MAX_FEATURE_BOOKS = max(20, int(os.getenv("MAX_FEATURE_BOOKS", "500")))
MAX_READERS = max(100, int(os.getenv("MAX_READERS", "100000")))
