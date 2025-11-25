import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from search_engine import SearchEngine


def precision_at_k(results: pd.DataFrame, true_category: str, k: int) -> float:
    top_k = results.head(k)
    if len(top_k) == 0:
        return 0.0
    correct = (top_k["category"] == true_category).sum()
    return correct / k


def recall_at_k(results: pd.DataFrame, true_category: str, k: int, df_all: pd.DataFrame) -> float:
    total_relevant = (df_all["category"] == true_category).sum()
    if total_relevant == 0:
        return 0.0
    top_k = results.head(k)
    correct = (top_k["category"] == true_category).sum()
    return correct / total_relevant


def average_precision(results: pd.DataFrame, true_category: str) -> float:
    """
    AP = average of precision@k at each rank where a relevant document appears.
    """
    if len(results) == 0:
        return 0.0

    hits = 0
    ap_sum = 0.0
    total_relevant = (results["category"] == true_category).sum()
    if total_relevant == 0:
        return 0.0

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        if row["category"] == true_category:
            hits += 1
            ap_sum += hits / rank

    return ap_sum / total_relevant


def ndcg(results: pd.DataFrame, true_category: str) -> float:
    """
    nDCG using:
      - relevance: 1 if category == true_category else 0
      - score: model score (used as ranking signal)
    """
    if len(results) == 0:
        return 0.0

    relevance = (results["category"] == true_category).astype(int).values
    scores = results["score"].values

    if relevance.sum() == 0:
        return 0.0

    return float(ndcg_score([relevance], [scores]))


def evaluate_single_query(engine: SearchEngine, query: str, true_category: str, k: int = 10) -> pd.DataFrame:
    """
    Evaluate both TF-IDF and BM25 for a single query + category.
    Returns a DataFrame with metrics for the two models.
    """

    # Get ranked results (no filters, full ranking)
    tfidf_results = engine.search_tfidf(query, top_k=100)
    bm25_results = engine.search_bm25(query, top_k=100)

    # TF-IDF metrics
    tfidf_p = precision_at_k(tfidf_results, true_category, k)
    tfidf_r = recall_at_k(tfidf_results, true_category, k, engine.df)
    tfidf_ap = average_precision(tfidf_results, true_category)
    tfidf_ndcg = ndcg(tfidf_results, true_category)

    # BM25 metrics
    bm25_p = precision_at_k(bm25_results, true_category, k)
    bm25_r = recall_at_k(bm25_results, true_category, k, engine.df)
    bm25_ap = average_precision(bm25_results, true_category)
    bm25_ndcg = ndcg(bm25_results, true_category)

    data = [
        {
            "model": "TF-IDF",
            "precision@k": tfidf_p,
            "recall@k": tfidf_r,
            "AP": tfidf_ap,
            "nDCG": tfidf_ndcg,
        },
        {
            "model": "BM25",
            "precision@k": bm25_p,
            "recall@k": bm25_r,
            "AP": bm25_ap,
            "nDCG": bm25_ndcg,
        },
    ]

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example manual test
    se = SearchEngine()
    q = "phishing email malicious link"
    true_cat = "Phishing"

    df_metrics = evaluate_single_query(se, q, true_cat, k=10)
    print("Dynamic evaluation for query:", q)
    print("True category:", true_cat)
    print(df_metrics)
