import numpy as np
import pandas as pd
from search_engine import SearchEngine
from sklearn.metrics import ndcg_score


EVALUATION_QUERIES = [
    {"query": "phishing email malicious link", "category": "Phishing"},
    {"query": "corporate ddos network attack", "category": "DDoS"},
    {"query": "ransomware network vulnerability", "category": "Ransomware"},
    {"query": "malware infection through web", "category": "Malware"},
    {"query": "data breach global threat", "category": "Data Breach"},
]


def precision_at_k(results, true_category, k):
    top_k = results.head(k)
    correct = (top_k["category"] == true_category).sum()
    return correct / k


def recall_at_k(results, true_category, k, df):
    total_relevant = (df["category"] == true_category).sum()
    if total_relevant == 0:
        return 0
    top_k = results.head(k)
    correct = (top_k["category"] == true_category).sum()
    return correct / total_relevant


def average_precision(results, true_category):
    """AP = sum over precision@k for each relevant doc / total relevant"""
    relevant_positions = []
    for idx, row in results.iterrows():
        if row["category"] == true_category:
            relevant_positions.append(idx)

    if len(relevant_positions) == 0:
        return 0.0

    ap = 0
    hits = 0
    for k, (_, row) in enumerate(results.iterrows(), start=1):
        if row["category"] == true_category:
            hits += 1
            ap += hits / k

    return ap / len(relevant_positions)


def evaluate_model(engine, model_name, top_k=10):
    metrics = []

    for item in EVALUATION_QUERIES:
        query = item["query"]
        category = item["category"]

        if model_name == "tfidf":
            results = engine.search_tfidf(query, top_k=100)
        else:
            results = engine.search_bm25(query, top_k=100)

        p_at_k = precision_at_k(results, category, top_k)
        r_at_k = recall_at_k(results, category, top_k, engine.df)
        ap = average_precision(results, category)

        # Prepare relevance vector for ndcg
        relevance = (results["category"] == category).astype(int).values
        ndcg = ndcg_score([relevance], [np.arange(len(relevance), 0, -1)])  # pseudo gains

        metrics.append({
            "query": query,
            "category": category,
            "precision@k": p_at_k,
            "recall@k": r_at_k,
            "AP": ap,
            "nDCG": ndcg,
        })

    return pd.DataFrame(metrics)


def main():
    print("[+] Initializing Search Engine for evaluation...")
    engine = SearchEngine()

    print("\n=== Evaluating TF-IDF ===")
    tfidf_metrics = evaluate_model(engine, "tfidf", top_k=10)
    print(tfidf_metrics)

    print("\n=== Evaluating BM25 ===")
    bm25_metrics = evaluate_model(engine, "bm25", top_k=10)
    print(bm25_metrics)

    # Save results
    tfidf_metrics.to_csv("../reports/tfidf_metrics.csv", index=False)
    bm25_metrics.to_csv("../reports/bm25_metrics.csv", index=False)

    print("\nSaved evaluation results to /reports/")


if __name__ == "__main__":
    main()
