import matplotlib.pyplot as plt
import pandas as pd

from search_engine import SearchEngine
from dynamic_evaluation import evaluate_single_query


# ---------- 1) VISUALIZE RESULTS OF ONE MODEL FOR A QUERY ----------

def visualize_results_for_query(engine: SearchEngine, query: str, model: str = "tfidf", top_k: int = 20):
    """
    Run a query with the chosen model and visualize the distribution
    of categories, attack vectors, severity, and (optionally) actors
    among the top_k retrieved documents.
    """
    if model.lower() == "tfidf":
        results = engine.search_tfidf(query, top_k=top_k)
        title_prefix = "TF-IDF"
    elif model.lower() == "bm25":
        results = engine.search_bm25(query, top_k=top_k)
        title_prefix = "BM25"
    else:
        results = engine.search_boolean(query)
        results = results.head(top_k)
        title_prefix = "Boolean"

    if results.empty:
        print("[!] No results for this query.")
        return

    print(f"[+] Retrieved {len(results)} documents using {title_prefix} for query: {query!r}")

    # Categories
    plt.figure(figsize=(6, 4))
    results["category"].value_counts().plot(kind="bar")
    plt.title(f"{title_prefix} – Category distribution (top {top_k})")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Attack vectors
    plt.figure(figsize=(6, 4))
    results["vector"].value_counts().plot(kind="bar")
    plt.title(f"{title_prefix} – Attack vector distribution (top {top_k})")
    plt.xlabel("Attack Vector")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Severity
    plt.figure(figsize=(6, 4))
    results["severity"].value_counts().sort_index().plot(kind="bar")
    plt.title(f"{title_prefix} – Severity distribution (top {top_k})")
    plt.xlabel("Severity")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Actors (top 10)
    plt.figure(figsize=(6, 4))
    results["actor"].value_counts().head(10).plot(kind="bar")
    plt.title(f"{title_prefix} – Top 10 actors (top {top_k})")
    plt.xlabel("Actor")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# ---------- 2) VISUALIZE COMPARISON BETWEEN MODELS FOR A QUERY ----------

def visualize_model_comparison_for_query(engine: SearchEngine, query: str, true_category: str, k: int = 10):
    """
    For a given query and its expected category, compute TF-IDF and BM25
    metrics dynamically and show a bar chart comparing them.
    """
    print(f"[+] Dynamic evaluation for query: {query!r}")
    print(f"    True category for relevance: {true_category!r}")
    metrics_df = evaluate_single_query(engine, query, true_category, k=k)
    print(metrics_df)

    # Bar chart comparison
    metrics = ["precision@k", "recall@k", "AP", "nDCG"]
    plot_df = metrics_df.set_index("model")[metrics]

    plt.figure(figsize=(7, 5))
    plot_df.plot(kind="bar")
    plt.title(f"Model comparison for query: {query}\n(true category: {true_category}, k={k})")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# ---------- MAIN: SIMPLE DEMO ----------

if __name__ == "__main__":
    se = SearchEngine()

    # Example query
    query = "phishing email malicious link"
    true_category = "Phishing"

    # 1) Visualize result distributions for TF-IDF
    visualize_results_for_query(se, query, model="tfidf", top_k=20)

    # 2) Visualize TF-IDF vs BM25 comparison for this query
    visualize_model_comparison_for_query(se, query, true_category, k=10)
