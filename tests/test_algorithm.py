import pandas as pd
from app.algorithm import run_kmeans, recommend

def dataset():
    books=pd.DataFrame({"book_id":[f"b{i}" for i in range(1,7)],"title":[f"Book {i}" for i in range(1,7)],"authors":["Author"]*6})
    rows=[]
    for u in range(1,13):
        for b in range(1,7):
            score=5 if ((u<=6 and b<=3) or (u>6 and b>=4)) else 2
            rows.append({"user_id":str(u),"book_id":f"b{b}","rating":score})
    return books,pd.DataFrame(rows)

def test_real_clustering_pipeline():
    b,r=dataset();out=run_kmeans(b,r,n_clusters=2,rating_threshold=3,max_feature_books=20)
    assert out["metrics"]["usable_readers"]==12
    assert len(out["clusters"])==2
    assert out["parameters"]["algorithm"] in {"kmeans","minibatch_kmeans"}

def test_cold_start_recommendations():
    b,r=dataset();out=run_kmeans(b,r,n_clusters=2,rating_threshold=3,max_feature_books=20)
    rec=recommend(out,"unknown-reader",5)
    assert rec["is_cold_start"] is True
