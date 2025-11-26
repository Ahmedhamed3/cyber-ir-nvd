import os
import re

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from wordcloud import WordCloud
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk

from search_engine import SearchEngine

nltk.download("stopwords")

CYBER_CATEGORIES = ["Malware", "Phishing", "Ransomware", "DDoS"]
SPORTS_CATEGORIES = [
    "Football",
    "Basketball",
    "Tennis",
    "Athletics",
    "Gymnastics",
    "Cycling",
    "Esports",
    "Swimming",
    "Handball",
    "Volleyball",
]
FOOD_CATEGORIES = ["Vegetable", "Fruit", "Grain", "Protein", "Dairy", "Nut", "Legume"]

stop_words = set(stopwords.words("english"))


# -------------------------------------------------------------------
# Cache SearchEngine
# -------------------------------------------------------------------
@st.cache_resource
def load_engine():
    return SearchEngine()


# -------------------------------------------------------------------
# Helper: highlight matched tokens in clean_text using HTML + CSS
# -------------------------------------------------------------------
def highlight_text(text: str, matched_tokens: set[str]) -> str:
    """
    Highlight matched tokens in the text using a colored <span>.
    Matching is case-insensitive and uses word boundaries.
    """
    if not text:
        return ""

    highlighted = text

    # Replace longer tokens first to avoid partial overlaps
    for tok in sorted(matched_tokens, key=len, reverse=True):
        if not tok:
            continue
        pattern = re.compile(rf"\b({re.escape(tok)})\b", flags=re.IGNORECASE)

        # Wrap each match in a span with a CSS class
        highlighted = pattern.sub(
            lambda m: f'<span class="hl-token">{m.group(1)}</span>',
            highlighted,
        )

    return highlighted


# -------------------------------------------------------------------
# Helper: compute IR metrics (score-based relevance, Option B)
# -------------------------------------------------------------------
def compute_ir_metrics_from_scores(
    scores_norm: np.ndarray,
    k: int,
    threshold: float,
) -> dict:
    """
    scores_norm: 1D numpy array of normalized scores in [0, 1]
    k: cutoff for Precision@k, Recall@k, nDCG@k
    threshold: relevance threshold (scores >= threshold are 'relevant')

    Returns a dict with precision@k, recall@k, AP, nDCG.
    """
    n_docs = len(scores_norm)
    if n_docs == 0:
        return {
            "precision@k": 0.0,
            "recall@k": 0.0,
            "AP": 0.0,
            "nDCG": 0.0,
        }

    # Binary relevance
    relevant_mask = scores_norm >= threshold
    total_rel = int(relevant_mask.sum())

    # Sort documents by score (descending)
    order = np.argsort(scores_norm)[::-1]
    rel_sorted = relevant_mask[order]

    # Cut k if bigger than number of docs
    k = min(k, n_docs)
    rel_k = rel_sorted[:k]
    num_rel_at_k = int(rel_k.sum())

    # Precision@k
    precision_k = num_rel_at_k / k if k > 0 else 0.0

    # Recall@k
    recall_k = num_rel_at_k / total_rel if total_rel > 0 else 0.0

    # Average Precision (AP) over the whole ranking
    if total_rel == 0:
        ap = 0.0
    else:
        hits = 0
        sum_prec = 0.0
        for i, is_rel in enumerate(rel_sorted, start=1):
            if is_rel:
                hits += 1
                sum_prec += hits / i
        ap = sum_prec / total_rel

    # nDCG@k
    def dcg(rels):
        return float(
            sum(rel / np.log2(idx + 2) for idx, rel in enumerate(rels))
        )

    rel_k_float = rel_k.astype(float)
    dcg_k = dcg(rel_k_float)

    if total_rel == 0:
        ndcg = 0.0
    else:
        ideal_rels = np.sort(relevant_mask.astype(float))[::-1][:k]
        idcg_k = dcg(ideal_rels)
        ndcg = dcg_k / idcg_k if idcg_k > 0 else 0.0

    return {
        "precision@k": precision_k,
        "recall@k": recall_k,
        "AP": ap,
        "nDCG": ndcg,
    }


def evaluate_query_score_based(
    se: SearchEngine,
    query: str,
    k: int,
    threshold: float,
) -> pd.DataFrame:
    """
    Option B: score-based relevance.

    For a given query, compute metrics for:
      - TF-IDF (VSM)
      - BM25

    Relevance is defined as:
      score_norm >= threshold

    For TF-IDF: scores = cosine similarity in [0, 1].
    For BM25: scores are normalized by max raw score in the corpus.
    """
    if not query.strip():
        raise ValueError("Query is empty.")

    # Preprocess query (same as search engine)
    clean_query = se._preprocess_query(query)
    if not clean_query:
        raise ValueError("Query became empty after preprocessing.")

    # ---------- TF-IDF ----------
    q_vec = se.tfidf_vectorizer.transform([clean_query])
    sims = cosine_similarity(q_vec, se.X_tfidf)[0]  # already in [0,1]
    scores_tfidf = sims.copy()
    metrics_tfidf = compute_ir_metrics_from_scores(
        scores_tfidf, k=k, threshold=threshold
    )

    # ---------- BM25 ----------
    tokens = clean_query.split()
    if tokens:
        bm25_raw = np.array(se.bm25.get_scores(tokens), dtype=float)
    else:
        bm25_raw = np.zeros(len(se.df), dtype=float)

    if bm25_raw.max() > 0:
        scores_bm25 = bm25_raw / bm25_raw.max()  # normalize
    else:
        scores_bm25 = bm25_raw

    metrics_bm25 = compute_ir_metrics_from_scores(
        scores_bm25, k=k, threshold=threshold
    )

    # Build DataFrame
    rows = []
    rows.append(
        {
            "model": "TF-IDF",
            "precision@k": metrics_tfidf["precision@k"],
            "recall@k": metrics_tfidf["recall@k"],
            "AP": metrics_tfidf["AP"],
            "nDCG": metrics_tfidf["nDCG"],
        }
    )
    rows.append(
        {
            "model": "BM25",
            "precision@k": metrics_bm25["precision@k"],
            "recall@k": metrics_bm25["recall@k"],
            "AP": metrics_bm25["AP"],
            "nDCG": metrics_bm25["nDCG"],
        }
    )

    return pd.DataFrame(rows)

    
def safe(val) -> str:
    return "" if pd.isna(val) or val == "" else str(val)

def format_result_header(row) -> str:
    category = safe(row.get("category", "")) or "Unknown"
    score = row.get("score", 0)
    try:
        score_str = f"{float(score):.4f}"
    except (TypeError, ValueError):
        score_str = "0.0000"

    if category in CYBER_CATEGORIES:
        topic_group = "Cybersecurity"
    elif category in SPORTS_CATEGORIES:
        topic_group = "Sports"
    elif category in FOOD_CATEGORIES:
        topic_group = "Food & Nutrition"
    else:
        topic_group = "Other"

    header_parts = []

    if topic_group == "Cybersecurity":
        actor = safe(row.get("actor", ""))
        vector = safe(row.get("vector", ""))
        location = safe(row.get("location", ""))
        severity_val = row.get("severity", "")
        severity = "" if pd.isna(severity_val) or severity_val == "" else str(severity_val)

        if actor:
            header_parts.append(actor)

        if vector or location:
            vector_segment = vector
            if location:
                vector_segment = f"{vector_segment} – {location}" if vector_segment else location
            if vector_segment:
                header_parts.append(vector_segment)

        if severity:
            header_parts.append(f"Severity: {severity}")

    elif topic_group == "Sports":
        team_or_player = safe(row.get("team_or_player", ""))
        event_type = safe(row.get("event_type", ""))
        location = safe(row.get("location", ""))

        if team_or_player:
            header_parts.append(team_or_player)

        if event_type or location:
            event_segment = event_type
            if location:
                event_segment = f"{event_segment} – {location}" if event_segment else location
            if event_segment:
                header_parts.append(event_segment)

    elif topic_group == "Food & Nutrition":
        location = safe(row.get("location", ""))
        if location:
            header_parts.append(location)

    else:
        location = safe(row.get("location", ""))
        if location:
            header_parts.append(location)

    header_parts.append(f"Score: {score_str}")

    return f"[{category}] " + " | ".join(header_parts)



# -------------------------------------------------------------------
# Main Streamlit app
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="IR System",
        layout="wide",
    )

    st.title("🔍 Information Retrieval System")

    st.markdown(
    """
    <style>
    .hl-token {
        background-color: #ffea00;
        color: #000;
        padding: 2px 5px;
        border-radius: 4px;
        font-weight: 700;
        box-shadow: 0 0 4px #ffea00;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

    se = load_engine()
    df = se.df

    tab_search, tab_dataset, tab_diag = st.tabs(
        [
            "Search + Dynamic Evaluation",
            "Dataset Analytics",
            "Model Diagnostics & Debugger",
        ]
    )

    # ===================== TAB 1: SEARCH + DYNAMIC EVAL =====================
    with tab_search:
       

        col_left, col_right = st.columns([2, 1])

        with col_left:
            query = st.text_input("Enter your search query:")

        with col_right:
            model = st.selectbox(
                "Retrieval model",
                ["TF-IDF (VSM)", "BM25", "Boolean"],
            )
            top_k = st.slider("Number of results (k)", 5, 50, 10, step=5)

        # Filters
        st.markdown("### Optional Filters")
        

        topic_group = st.selectbox(
            "Topic Group",
            ["All", "Cybersecurity", "Sports", "Food & Nutrition"],
            index=0,
        )

        if topic_group == "All":
            topic_df = df
            category_options = ["All"] + sorted(df["category"].dropna().unique().tolist())
        elif topic_group == "Cybersecurity":
            topic_df = df[df["category"].isin(CYBER_CATEGORIES)]
            category_options = ["All"] + CYBER_CATEGORIES
        elif topic_group == "Sports":
            topic_df = df[df["category"].isin(SPORTS_CATEGORIES)]
            category_options = ["All"] + SPORTS_CATEGORIES
        else:  # Food & Nutrition
            topic_df = df[df["category"].isin(FOOD_CATEGORIES)]
            category_options = ["All"] + FOOD_CATEGORIES
        category_filter = "All"
        actor_filter = "All"
        vector_filter = "All"
        location_filter = "All"
        severity_filter = "All"
        team_player_filter = "All"
        event_type_filter = "All"

        if topic_group == "Cybersecurity":
            f1, f2, f3, f4, f5 = st.columns(5)

            with f1:
                category_filter = st.selectbox(
                    "Category",
                    category_options,
                    index=0,
                )
            with f2:
                actor_filter = st.selectbox(
                    "Actor",
                    ["All"] + sorted(topic_df["actor"].dropna().unique().tolist()),
                    index=0,
                )
            with f3:
                vector_filter = st.selectbox(
                    "Vector",
                    ["All"] + sorted(topic_df["vector"].dropna().unique().tolist()),
                    index=0,
                )
            with f4:
                location_filter = st.selectbox(
                    "Location",
                    ["All"] + sorted(topic_df["location"].dropna().unique().tolist()),
                    index=0,
                )
            with f5:
                severity_filter = st.selectbox(
                    "Severity",
                    ["All"]
                    + sorted(topic_df["severity"].dropna().astype(str).unique().tolist()),
                    index=0,
                )
        elif topic_group == "Sports":
            f1, f2, f3, f4 = st.columns(4)

            with f1:
                category_filter = st.selectbox("Category", category_options, index=0)
            with f2:
                team_player_filter = st.selectbox(
                    "Team/Player",
                    ["All"]
                    + sorted(
                        topic_df["team_or_player"].dropna().astype(str).unique().tolist()
                    ),
                    index=0,
                )
            with f3:
                event_type_filter = st.selectbox(
                    "Event Type",
                    ["All"]
                    + sorted(
                        topic_df["event_type"].dropna().astype(str).unique().tolist()
                    ),
                    index=0,
                )
            with f4:
                location_filter = st.selectbox(
                    "Location",
                    ["All"] + sorted(topic_df["location"].dropna().unique().tolist()),
                    index=0,
                )
        else:  # Food & Nutrition
            f1, f2 = st.columns(2)

            with f1:
                category_filter = st.selectbox("Category", category_options, index=0)
            with f2:
                location_filter = st.selectbox(
                    "Location",
                    ["All"] + sorted(topic_df["location"].dropna().unique().tolist()),
                    index=0,
                )

        filtered_df = topic_df.copy()

        if category_filter != "All":
            filtered_df = filtered_df[filtered_df["category"] == category_filter]

        if topic_group == "Cybersecurity":
            if actor_filter != "All":
                filtered_df = filtered_df[
                    filtered_df["actor"].notna() & (filtered_df["actor"] == actor_filter)
                ]
            if vector_filter != "All":
                filtered_df = filtered_df[
                    filtered_df["vector"].notna()
                    & (filtered_df["vector"] == vector_filter)
                ]
            if severity_filter != "All":
                filtered_df = filtered_df[
                    filtered_df["severity"].notna()
                    & (filtered_df["severity"].astype(str) == severity_filter)
                ]
        elif topic_group == "Sports":
            if team_player_filter != "All":
                filtered_df = filtered_df[
                    filtered_df["team_or_player"].notna()
                    & (filtered_df["team_or_player"].astype(str) == team_player_filter)
                ]
            if event_type_filter != "All":
                filtered_df = filtered_df[
                    filtered_df["event_type"].notna()
                    & (filtered_df["event_type"].astype(str) == event_type_filter)
                ]

        if location_filter != "All":
            filtered_df = filtered_df[
                filtered_df["location"].notna() & (filtered_df["location"] == location_filter)
            ]

        allowed_indices = filtered_df.index

        search_btn = st.button("Search")

        results = None
        has_valid_results = False

        if search_btn:
            if filtered_df.empty:
                st.info("No documents match the selected filters.")
            elif not query.strip():
                st.warning("Please enter a query.")
            else:
                try:
                    # Preprocess query using the same logic as SearchEngine
                    query_clean = se._preprocess_query(query)
                    query_tokens = set(query_clean.split())

                    search_pool = len(df)

                    # --- Run selected model ---
                    if model == "TF-IDF (VSM)":
                        results = se.search_tfidf(
                            query, top_k=search_pool, filters=None
                        )
                    elif model == "BM25":
                        results = se.search_bm25(query, top_k=search_pool, filters=None)
                    else:
                        results = se.search_boolean(query, filters=None)
                        if len(results) > search_pool:
                            results = results.head(search_pool)

                    if results is not None:
                        results = results[results.index.isin(allowed_indices)]
                        results = results.head(top_k)


                    if results is None or results.empty:
                        st.warning("No documents matched your query and filters.")
                    else:
                        if "score" in results.columns:
                            score_series = results["score"]
                            if score_series.isna().all() or (score_series.fillna(0) == 0).all():
                                st.warning("No documents matched your query and filters.")
                                results = None
                            else:
                                has_valid_results = True
                        else:
                            has_valid_results = True

                        if has_valid_results and results is not None:
                            st.success(
                                f"Found {len(results)} results (showing up to {top_k})."
                            )

                            for idx, row in results.iterrows():
                                clean_text = str(row.get("clean_text", ""))
                                doc_tokens = set(clean_text.lower().split())
                                matched_tokens = doc_tokens & query_tokens

                                highlighted = highlight_text(clean_text, matched_tokens)

                                header = format_result_header(row)

                                with st.expander(header):
                                    # ===== Description with highlighting =====
                                    st.markdown("**Description (processed `clean_text`):**")
                                    st.markdown(highlighted, unsafe_allow_html=True)

                                    # ===== Explainability panel =====
                                    st.markdown("**Why is this document ranked here?**")
                                    st.write(f"- Retrieval model: `{model}`")
                                    st.write(f"- Score (normalized): `{row.get('score', 0):.4f}`")

                                    if model == "BM25":
                                        bm25_raw_val = row.get("bm25_raw", None)
                                        if bm25_raw_val is not None:
                                            st.write(f"- BM25 raw score: `{bm25_raw_val:.4f}`")

                                    if query_tokens:
                                        st.write(
                                            f"- Matched tokens: "
                                            f"({len(matched_tokens)}/{len(query_tokens)}) "
                                            + (
                                                ", ".join(sorted(matched_tokens))
                                                if matched_tokens
                                                else "None"
                                            )

                                      )
                                    else:
                                        st.write(
                                            "- Matched tokens: query had no valid tokens after preprocessing."
                                        )

                                    

                except Exception as e:
                    st.error(f"Error during search: {e}")

        # ========== Dynamic Evaluation (Option B: score-based) ==========
        if has_valid_results:
            st.markdown("---")
            st.subheader("⚖️ Dynamic Evaluation for This Query (Score-based Relevance)")

        col_eval1, col_eval2, col_eval3 = st.columns([2, 1, 1])

        with col_eval1:
                st.write(
                    "Relevance is defined as: **documents whose normalized score ≥ threshold**.\n"
                    "For TF-IDF, scores are cosine similarities in [0, 1].\n"
                    "For BM25, scores are normalized by the maximum raw score."
                )

        with col_eval2:
                k_eval = st.slider(
                    "k for evaluation (Precision@k, Recall@k, nDCG@k)",
                    5,
                    50,
                    10,
                    step=5,
                )

        with col_eval3:
                relevance_threshold = st.slider(
                    "Score threshold for relevance",
                    0.0,
                    1.0,
                    0.2,
                    step=0.05,
                )

        decision_metric_ui = st.selectbox(
                "Metric for recommendation",
                ["nDCG", "AP", "precision@k", "recall@k"],
                index=0,
            )


        if st.button("Evaluate this query"):
                if not query.strip():
                    st.warning("Enter a query first.")
                else:
                    try:
                        metrics_df = evaluate_query_score_based(
                            se, query, k=k_eval, threshold=relevance_threshold
                         )
                    

                        st.write("Dynamic evaluation for this query (score-based relevance):")
                        st.dataframe(
                            metrics_df.style.format(
                                {
                                    "precision@k": "{:.3f}",
                                    "recall@k": "{:.3f}",
                                    "AP": "{:.3f}",
                                    "nDCG": "{:.3f}",
                                }
                            )
                        )    
                    

                         # ===== Recommendation based on selected metric =====
                        decision_metric = decision_metric_ui

                        if decision_metric not in metrics_df.columns:
                            st.error(
                                f"Selected metric '{decision_metric}' not found in metrics table."
                            )
                        else:
                            best_idx = metrics_df[decision_metric].idxmax()
                            best_row = metrics_df.loc[best_idx]
                            best_model = best_row["model"]
                            best_score = float(best_row[decision_metric])

                            other_rows = metrics_df[metrics_df["model"] != best_model]
                            if not other_rows.empty:
                                other_score = float(other_rows[decision_metric].iloc[0])
                            else:
                                other_score = best_score

                            diff = abs(best_score - other_score)

                            st.markdown("### Model Recommendation")

                            if diff < 0.02:
                                st.info(
                                    f"For this query, **TF-IDF** and **BM25** perform very similarly "
                                    f"based on `{decision_metric}` "
                                    f"({best_score:.3f} vs {other_score:.3f}). "
                                    f"You can use either model."
                                )
                            else:
                                st.success(
                                    f"For this query, **`{best_model}`** is recommended, "
                                    f"as it achieves a higher `{decision_metric}` "
                                    f"({best_score:.3f} vs {other_score:.3f})."
                                )

                        st.markdown("### Metrics comparison (bar chart)")
                        chart_df = metrics_df.set_index("model")[
                            ["precision@k", "recall@k", "AP", "nDCG"]
                        ]
                        st.bar_chart(chart_df)

                    except Exception as e:
                        st.error(f"Error during dynamic evaluation: {e}")

# ========== Result insights for current results ==========
        if has_valid_results and results is not None:
            st.markdown("---")
            st.subheader("🧭 Result Insights")

            # Guard clause: show message when no rows are available
            if results.empty:
                st.info(
                    "No insights to display because there are no matching results."
                )
            else:
               

                total_results = len(results)

                # KPI cards for quick overview
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)

                with kpi1:
                    st.markdown("**Total Results**")
                    st.markdown(f"<h3 style='margin-top: -5px'>{total_results}</h3>", unsafe_allow_html=True)

                with kpi2:
                    cyber_count = results["category"].isin(CYBER_CATEGORIES).sum()
                    st.markdown("**Cybersecurity Docs**")
                    st.markdown(
                        f"<h3 style='margin-top: -5px'>{cyber_count}</h3>",
                        unsafe_allow_html=True,
                    )

                with kpi3:
                    sports_count = results["category"].isin(SPORTS_CATEGORIES).sum()
                    st.markdown("**Sports Docs**")
                    st.markdown(
                        f"<h3 style='margin-top: -5px'>{sports_count}</h3>",
                        unsafe_allow_html=True,
                    )

                with kpi4:
                    food_count = results["category"].isin(FOOD_CATEGORIES).sum()
                    st.markdown("**Food & Nutrition Docs**")
                    st.markdown(
                        f"<h3 style='margin-top: -5px'>{food_count}</h3>",
                        unsafe_allow_html=True,
                    )

                # Topic breakdown table
                st.markdown("### Topic Breakdown")

                def map_topic_group(cat: str) -> str:
                    if cat in CYBER_CATEGORIES:
                        return "Cybersecurity"
                    if cat in SPORTS_CATEGORIES:
                        return "Sports"
                    if cat in FOOD_CATEGORIES:
                        return "Food & Nutrition"
                    return "Other"

                category_counts = results["category"].fillna("Unknown").value_counts()
                breakdown_rows = []
                for category_value, count in category_counts.items():
                    topic_group = map_topic_group(category_value)
                    percentage = round((count / total_results) * 100, 1)
                    breakdown_rows.append(
                        {
                            "Topic Group": topic_group,
                            "Category": category_value,
                            "Count": count,
                            "Percentage": percentage,
                        }
                    )

                breakdown_df = pd.DataFrame(breakdown_rows)
                breakdown_df = breakdown_df.sort_values(by="Count", ascending=False)

                st.dataframe(breakdown_df, width="stretch")

                # Key Entities area (tables only, topic-aware)
                st.markdown("### Key Entities")
                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown("#### Cybersecurity Entities")
                    if cyber_count == 0:
                        st.info("No cybersecurity documents in the current results.")
                    else:
                        cyber_df = results[results["category"].isin(CYBER_CATEGORIES)]

                        # Top actors
                        actors_series = cyber_df["actor"].dropna().astype(str).str.strip()
                        actors_series = actors_series[actors_series != ""]
                        actor_counts = actors_series.value_counts().head(5)
                        if not actor_counts.empty:
                            st.write("Top Actors")
                            st.table(actor_counts.reset_index().rename(columns={"index": "Actor", "actor": "Count"}))
                        else:
                            st.info("No actor information available.")

                        # Top attack vectors
                        vector_series = cyber_df["vector"].dropna().astype(str).str.strip()
                        vector_series = vector_series[vector_series != ""]
                        vector_counts = vector_series.value_counts().head(5)
                        if not vector_counts.empty:
                            st.write("Top Attack Vectors")
                            st.table(
                                vector_counts.reset_index().rename(
                                    columns={"index": "Vector", "vector": "Count"}
                                )
                            )
                        else:
                            st.info("No attack vector information available.")

                with col_right:
                    st.markdown("#### Sports Entities")
                    if sports_count == 0:
                        st.info("No sports documents in the current results.")
                    else:
                        sports_df = results[results["category"].isin(SPORTS_CATEGORIES)]

                        # Top teams/players
                        team_series = sports_df["team_or_player"].dropna().astype(str).str.strip()
                        team_series = team_series[team_series != ""]
                        team_counts = team_series.value_counts().head(5)
                        if not team_counts.empty:
                            st.write("Top Teams / Players")
                            st.table(
                                team_counts.reset_index().rename(
                                    columns={"index": "Team/Player", "team_or_player": "Count"}
                                )
                            )
                        else:
                            st.info("No team/player information available.")

                        # Top event types
                        event_series = sports_df["event_type"].dropna().astype(str).str.strip()
                        event_series = event_series[event_series != ""]
                        event_counts = event_series.value_counts().head(5)
                        if not event_counts.empty:
                            st.write("Top Event Types")
                            st.table(
                                event_counts.reset_index().rename(
                                    columns={"index": "Event Type", "event_type": "Count"}
                                )
                            )
                        else:
                            st.info("No event type information available.")

    # ===================== TAB 2: DATASET ANALYTICS =====================
    with tab_dataset:
        st.subheader("Dataset Overview")
        st.write("Analytics and key insights computed over the entire dataset.")

        total_docs = int(len(df))

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown("**Total documents**")
            st.markdown(
                f"<h3 style='margin-top: -5px'>{total_docs}</h3>",
                unsafe_allow_html=True,
            )

        with kpi2:
            cyber_docs = df["category"].isin(CYBER_CATEGORIES).sum()
            st.markdown("**Cybersecurity docs**")
            st.markdown(
                f"<h3 style='margin-top: -5px'>{cyber_docs}</h3>",
                unsafe_allow_html=True,
            )

        with kpi3:
            sports_docs = df["category"].isin(SPORTS_CATEGORIES).sum()
            st.markdown("**Sports docs**")
            st.markdown(
                f"<h3 style='margin-top: -5px'>{sports_docs}</h3>",
                unsafe_allow_html=True,
            )

        with kpi4:
            food_docs = df["category"].isin(FOOD_CATEGORIES).sum()
            st.markdown("**Food & Nutrition docs**")
            st.markdown(
                f"<h3 style='margin-top: -5px'>{food_docs}</h3>",
                unsafe_allow_html=True,
            )

        def map_topic_group(cat: str) -> str:
            if not cat:
                return None
            if cat in CYBER_CATEGORIES:
                return "Cybersecurity"
            if cat in SPORTS_CATEGORIES:
                return "Sports"
            if cat in FOOD_CATEGORIES:
                return "Food & Nutrition"
            return "Other"

        category_series = df["category"].astype(str).str.strip()
        category_series = category_series.replace({"": pd.NA, "nan": pd.NA})
        df["topic_group"] = category_series.apply(map_topic_group)

        topic_counts = df["topic_group"].dropna().value_counts()

        st.markdown("### Topic Group Distribution")

        if topic_counts.empty:
            st.info("No data available to plot topic distribution.")
        else:
            pie_df = (
                topic_counts.rename_axis("Topic Group")
                .reset_index(name="Count")
                .loc[:, ["Topic Group", "Count"]]
            )
            pie_df = pie_df[pie_df["Topic Group"].notna()]
            pie_df["Count"] = pd.to_numeric(pie_df["Count"], errors="coerce").fillna(0)

            if total_docs > 0:
                pie_df["percent"] = pie_df["Count"] / float(total_docs)
            else:
                pie_df["percent"] = 0.0

            chart = (
                alt.Chart(pie_df)
                .mark_arc(innerRadius=60)
                .encode(
                    theta=alt.Theta("Count:Q"),
                    color=alt.Color("Topic Group:N"),
                    tooltip=[
                        "Topic Group:N",
                        "Count:Q",
                        alt.Tooltip("percent:Q", format=".1%"),
                    ],
                )
            )

            st.altair_chart(chart, width="stretch")

            st.subheader("Word Cloud (Most Frequent Terms)")
        if df.empty:
            st.info("Dataset is empty; no text available to generate word cloud.")
        elif st.button("Generate Word Cloud"):
            all_text = " ".join(df["clean_text"].dropna().astype(str))
            if all_text.strip():
                wc = WordCloud(
                    width=800, height=400, background_color="white"
                ).generate(all_text)
                st.image(wc.to_array(), use_column_width=True)
            else:
                st.info("No text available to generate word cloud.")

                st.subheader("Word Frequency Chart (Top 20 Words)")
        if st.button("Show Word Frequency"):
            clean_series = df["clean_text"].dropna().astype(str)
            if clean_series.empty:
                st.info("No data available to compute word frequency.")
            else:
                all_text = " ".join(clean_series)
                words = [
                    w.lower()
                    for w in all_text.split()
                    if w.lower() not in stop_words and w.isalpha() and len(w) > 2
                ]
                if not words:
                    st.info("No data available to compute word frequency.")
                else:
                    freq = Counter(words)
                    top_words = freq.most_common(20)

                    if not top_words:
                        st.info("No frequent words found in the dataset.")
                    else:
                        freq_df = pd.DataFrame(top_words, columns=["Word", "Count"])

                        chart = (
                            alt.Chart(freq_df)
                            .mark_bar()
                            .encode(
                                x="Count:Q",
                                y=alt.Y("Word:N", sort="-x"),
                                tooltip=["Word", "Count"],
                                color=alt.value("#4C78A8"),
                            )
                        )

                        st.altair_chart(chart, width="stretch")
        st.markdown("#### Category Breakdown")

        breakdown_rows = []
        if total_docs > 0:
            category_counts = df["category"].dropna().value_counts()
            for category_value, count in category_counts.items():
                topic_group = map_topic_group(category_value)
                percentage = round((count / total_docs) * 100, 1)
                breakdown_rows.append(
                    {
                        "Topic Group": topic_group,
                        "Category": category_value,
                        "Count": count,
                        "Percentage": percentage,
                    }
                )

        breakdown_df = pd.DataFrame(
            breakdown_rows, columns=["Topic Group", "Category", "Count", "Percentage"]
        )
        if not breakdown_df.empty:
            breakdown_df = breakdown_df.sort_values(by="Count", ascending=False)

            st.dataframe(breakdown_df, width="stretch")

    # ===================== TAB 3: MODEL DIAGNOSTICS & DEBUGGER =====================
    with tab_diag:
        st.subheader("🧠 Model Diagnostics & Query Debugger")

        if not df.empty:
            st.markdown("### Corpus Overview")
            st.write(f"- Total documents: **{len(df)}**")
            avg_len = df["clean_text"].astype(str).apply(
                lambda x: len(x.split())
            ).mean()
            st.write(f"- Average document length (tokens): **{avg_len:.1f}**")
            st.write(
                f"- TF-IDF vocabulary size: **{len(se.tfidf_vectorizer.vocabulary_)}**"
            )

        if not query.strip():
            st.info(
                "Enter a query in the **Search + Dynamic Evaluation** tab to see diagnostics here."
            )
        else:
            clean_query = se._preprocess_query(query)
            tokens = clean_query.split()

            st.markdown("### Query Overview")
            st.write(f"- Raw query: `{query}`")
            st.write(f"- Preprocessed query: `{clean_query}`")
            st.write(f"- Number of tokens after preprocessing: **{len(tokens)}**")

            vocab = se.tfidf_vectorizer.vocabulary_
            known_tokens = [t for t in tokens if t in vocab]
            unknown_tokens = [t for t in tokens if t not in vocab]

            col_q1, col_q2 = st.columns(2)
            with col_q1:
                st.write("**Known tokens (in TF-IDF vocabulary):**")
                if known_tokens:
                    st.write(", ".join(known_tokens))
                else:
                    st.write("_None_")

            with col_q2:
                st.write("**Out-of-vocabulary tokens:**")
                if unknown_tokens:
                    st.write(", ".join(unknown_tokens))
                else:
                    st.write("_None_")

            coverage = (len(known_tokens) / len(tokens)) if tokens else 0.0
            st.write(f"- Vocabulary coverage: **{coverage * 100:.1f}%**")

            # ---------- TF-IDF Diagnostics ----------
            st.markdown("### TF-IDF Diagnostics for This Query")

            if tokens:
                q_vec = se.tfidf_vectorizer.transform([clean_query])
                sims = cosine_similarity(q_vec, se.X_tfidf)[0]

                st.write(
                    f"- Cosine similarity range: "
                    f"min = {sims.min():.4f}, max = {sims.max():.4f}, mean = {sims.mean():.4f}"
                )

                top_idx = np.argsort(sims)[::-1][:20]
                tfidf_diag_df = df.iloc[top_idx].copy()
                tfidf_diag_df["tfidf_score"] = sims[top_idx]

                diag_chart_df = tfidf_diag_df[["clean_text", "tfidf_score"]].copy()
                diag_chart_df = diag_chart_df.set_index(
                    diag_chart_df["clean_text"].str.slice(0, 40) + "..."
                )[["tfidf_score"]]

                st.write("Top documents by TF-IDF score:")
                st.bar_chart(diag_chart_df)
            else:
                st.info(
                    "Query has no valid tokens after preprocessing; TF-IDF diagnostics are not available."
                )

            # ---------- BM25 Diagnostics ----------
            st.markdown("### BM25 Diagnostics for This Query")

            if tokens:
                bm25_raw = np.array(se.bm25.get_scores(tokens), dtype=float)
                if bm25_raw.size > 0:
                    st.write(
                        f"- BM25 raw score range: "
                        f"min = {bm25_raw.min():.4f}, max = {bm25_raw.max():.4f}, "
                        f"mean = {bm25_raw.mean():.4f}"
                    )

                    top_idx_bm = np.argsort(bm25_raw)[::-1][:20]
                    bm25_diag_df = df.iloc[top_idx_bm].copy()
                    bm25_diag_df["bm25_raw"] = bm25_raw[top_idx_bm]

                    bm25_chart_df = bm25_diag_df[["clean_text", "bm25_raw"]].copy()
                    bm25_chart_df = bm25_chart_df.set_index(
                        bm25_chart_df["clean_text"].str.slice(0, 40) + "..."
                    )[["bm25_raw"]]

                    st.write("Top documents by BM25 raw score:")
                    st.bar_chart(bm25_chart_df)
            else:
                st.info(
                    "Query has no valid tokens after preprocessing; BM25 diagnostics are not available."
                )

            # ---------- Query Difficulty Heuristic ----------
            st.markdown("### Query Difficulty Assessment")

            messages = []
            if len(tokens) <= 1:
                messages.append(
                    "- The query is very short. Short queries often have ambiguous meaning."
                )
            if coverage < 0.5:
                messages.append(
                    "- Less than 50% of query tokens are in the vocabulary. "
                    "Consider using more common or technical terms from the dataset."
                )
            if not messages:
                messages.append(
                    "- The query seems reasonably well-formed for this corpus. "
                    "Both TF-IDF and BM25 should be able to retrieve useful results."
                )

            for m in messages:
                st.write(m)
                st.markdown("---")
        st.markdown("### Dataset Previews")

        preprocessed_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "processed", "ir_preprocessed_dataset.csv"
        )
        raw_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "raw", "Cybersecurity_Dataset.csv"
        )

        if st.button("Show Preprocessed Dataset"):
            try:
                preprocessed_df = pd.read_csv(preprocessed_path)
                st.dataframe(preprocessed_df)
            except FileNotFoundError:
                st.error("File not found")

        if st.button("Show Raw Cybersecurity Dataset"):
            try:
                raw_df = pd.read_csv(raw_path)
                st.dataframe(raw_df)
            except FileNotFoundError:
                st.error("File not found")


if __name__ == "__main__":
    main()
