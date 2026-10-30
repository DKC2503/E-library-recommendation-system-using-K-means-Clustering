import asyncio, logging, os, tempfile, uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from app.algorithm import run_kmeans, recommend
from app import database
from app.config import MAX_UPLOAD_MB, MAX_FEATURE_BOOKS, MAX_READERS

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] [elibrary] %(message)s")
log=logging.getLogger("elibrary")
app=FastAPI(title="E-Library K-Means Recommendation System",version="1.0")
app.mount("/static",StaticFiles(directory=str(Path(__file__).parent/"static")),name="static")
CURRENT=None; RUN_ID=None; HISTORY=[]; LOCK=asyncio.Lock()

@app.on_event("startup")
def startup(): database.connect()
@app.on_event("shutdown")
def shutdown(): database.close()
@app.get("/")
def home(): return FileResponse(Path(__file__).parent/"static"/"index.html")
@app.get("/api/health")
def health(): return {"status":"ok","algorithm":"scikit-learn K-Means + TruncatedSVD","mongodb":database.health()}
@app.get("/api/dataset/status")
def dataset_status():
    if CURRENT is not None:
        return {"ready":True,"run_id":RUN_ID,"dataset_id":CURRENT["dataset_id"],"metrics":CURRENT["metrics"],"parameters":CURRENT["parameters"],"source":"memory"}
    saved = database.latest_run()
    if saved:
        return {"ready":True,"run_id":saved.get("run_id"),"dataset_id":saved.get("dataset_id"),"metrics":saved.get("metrics",{}),"parameters":saved.get("parameters",{}),"source":"MongoDB"}
    return {"ready":False,"message":"No analysis is available. Upload both CSVs and run analysis."}
@app.get("/api/clusters")
def clusters():
    if CURRENT is not None:
        return {"run_id":RUN_ID,"clusters":CURRENT["clusters"],"metrics":CURRENT["metrics"],"parameters":CURRENT["parameters"]}
    db = database.get_db()
    saved = database.latest_run()
    if db is not None and saved:
        rows = list(db.cluster_summaries.find({"run_id":saved.get("run_id")},{"_id":0,"run_id":0}))
        if rows:
            return {"run_id":saved.get("run_id"),"clusters":rows,"metrics":saved.get("metrics",{}),"parameters":saved.get("parameters",{}),"source":"MongoDB"}
    raise HTTPException(409,"No saved analysis is available. Upload the CSV files and run clustering first.")
@app.get("/api/recommend/{reader_id}")
def recommendations(reader_id:str,limit:int=10):
    if not 1 <= limit <= 50: raise HTTPException(422,"limit must be between 1 and 50.")
    if CURRENT is not None:
        return recommend(CURRENT,reader_id,limit)
    try:
        saved = database.load_recommendations(reader_id,limit)
    except Exception as e:
        log.exception("Persisted recommendation lookup failed")
        raise HTTPException(503,"Saved recommendation data could not be read from MongoDB. Check the database connection and try again.") from e
    if saved is not None:
        return saved
    db_health = database.health()
    if db_health.get("connected"):
        raise HTTPException(409,"No persisted analysis is available yet. Upload the CSV files and run clustering first.")
    raise HTTPException(503,"No analysis is loaded and MongoDB persistence is unavailable. Run an analysis first or restore the database connection.")
@app.get("/api/history")
def history():
    db=database.get_db()
    if db is not None:
        try:
            rows=list(db.analysis_runs.find({},{"_id":0}).sort("created_at",-1).limit(20))
            if rows:return {"history":rows}
        except Exception as e: log.warning("Mongo history read failed: %s",e)
    return {"history":list(reversed(HISTORY[-20:]))}
@app.post("/api/analyze")
async def analyze(books_file:UploadFile=File(...),ratings_file:UploadFile=File(...),
    n_clusters:int=Form(5),rating_threshold:float=Form(3.0),algorithm:str=Form("minibatch_kmeans"),random_state:int=Form(42)):
    global CURRENT,RUN_ID
    if LOCK.locked(): raise HTTPException(429,"An analysis is already running.")
    if not (books_file.filename or "").lower().endswith(".csv") or not (ratings_file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400,"Both files must be CSV.")
    if not 2<=n_clusters<=50: raise HTTPException(400,"Cluster count must be between 2 and 50.")
    if not 0<=rating_threshold<=5: raise HTTPException(400,"Rating threshold must be between 0 and 5.")
    paths=[]
    try:
        async with LOCK:
            max_bytes=MAX_UPLOAD_MB*1024*1024
            for up in (books_file,ratings_file):
                fd,p=tempfile.mkstemp(prefix="elibrary_",suffix=".csv");os.close(fd);paths.append(p)
                size=0
                with open(p,"wb") as out:
                    while True:
                        chunk=await up.read(1024*1024)
                        if not chunk:break
                        size+=len(chunk)
                        if size>max_bytes:raise HTTPException(413,f"{up.filename} exceeds {MAX_UPLOAD_MB} MiB.")
                        out.write(chunk)
                if not size:raise HTTPException(400,f"{up.filename} is empty.")
            try: books=pd.read_csv(paths[0],low_memory=False); ratings=pd.read_csv(paths[1],low_memory=False)
            except Exception as e: raise HTTPException(400,f"Could not parse CSV: {e}")
            try:
                result=await asyncio.to_thread(run_kmeans,books,ratings,n_clusters,rating_threshold,algorithm,random_state,MAX_FEATURE_BOOKS,MAX_READERS)
            except ValueError as e:raise HTTPException(422,str(e))
            run_id="run_"+uuid.uuid4().hex[:12]
            public={k:v for k,v in result.items() if not k.startswith("_")}
            public.update({"run_id":run_id,"created_at":datetime.now(timezone.utc).isoformat(),"status":"completed"})
            CURRENT=result;RUN_ID=run_id;HISTORY.append(public)
            db=database.get_db()
            if db is not None:
                try:
                    database.persist_analysis(result, public)
                    log.info("Persisted model metadata, reader assignments, reader histories, catalogue and recommendation aggregates.")
                except Exception as e:
                    log.exception("Mongo persistence incomplete; current in-memory analysis remains usable: %s",e)
            log.info("Completed %s: %s",run_id,result["metrics"])
            return public
    finally:
        for p in paths:
            try:os.remove(p)
            except OSError:pass
        await books_file.close();await ratings_file.close()
