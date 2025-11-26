import os
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from search_engine import SearchEngine


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


# -------------------------------------------------------------------
# Main Streamlit app
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Cyber Threat IR System",
        layout="wide",
    )

    st.title("🔍 Cybersecurity Threat Information Retrieval System")

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
        st.subheader("Search Threat Descriptions")

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
        cyber_categories = ["Malware", "Phishing", "Ransomware", "DDoS"]
        sports_categories = [
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
        food_categories = ["Vegetable", "Fruit", "Grain", "Protein", "Dairy", "Nut", "Legume"]

        topic_group = st.selectbox(
            "Topic Group",
            ["All", "Cybersecurity", "Sports", "Food & Nutrition"],
            index=0,
        )

        if topic_group == "All":
            topic_df = df
            category_options = ["All"] + sorted(df["category"].dropna().unique().tolist())
        elif topic_group == "Cybersecurity":
            topic_df = df[df["category"].isin(cyber_categories)]
            category_options = ["All"] + cyber_categories
        elif topic_group == "Sports":
            topic_df = df[df["category"].isin(sports_categories)]
            category_options = ["All"] + sports_categories
        else:  # Food & Nutrition
            topic_df = df[df["category"].isin(food_categories)]
            category_options = ["All"] + food_categories

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

                                with st.expander(
                                    f"[{row.get('category', 'N/A')}] "
                                    f"{row.get('actor', 'Unknown')} | "
                                    f"{row.get('vector', 'Unknown')} – {row.get('location', 'Unknown')} | "
                                    f"Severity: {row.get('severity', 'N/A')} | "
                                    f"Score: {row.get('score', 0):.4f}"
                                ):
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

                                    # Similar threats using TF-IDF similarity
                                    if st.button("Show similar threats", key=f"sim_{idx}"):
                                        try:
                                            similar_docs = se.get_similar(idx, top_k=5)
                                            st.write("Top similar threats:")
                                            for j, srow in similar_docs.iterrows():
                                                st.markdown(
                                                    f"- [{srow.get('category', 'N/A')}] "
                                                    f"{srow.get('actor', 'Unknown')} | "
                                                    f"{srow.get('vector', 'Unknown')} – "
                                                    f"{srow.get('location', 'Unknown')} | "
                                                    f"Severity: {srow.get('severity', 'N/A')}"
                                                )
                                        except Exception as e:
                                            st.error(f"Error getting similar documents: {e}")

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

# ========== Analytics for current results ==========
        if has_valid_results and results is not None and not results.empty:
            st.markdown("---")
            st.subheader("📊 Analytics on Current Results")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.write("Categories")
                st.bar_chart(results["category"].value_counts())

            with c2:
                st.write("Attack Vectors")
                st.bar_chart(results["vector"].value_counts())

            with c3:
                st.write("Severity")
                st.bar_chart(results["severity"].value_counts().sort_index())

            with c4:
                st.write("Top Actors")
                st.bar_chart(results["actor"].value_counts().head(10))

            # Extra BM25 debug chart if BM25 is selected
            if model == "BM25" and "bm25_raw" in results.columns:
                st.markdown("---")
                st.subheader("🧪 BM25 Debug: Raw Score Distribution (Current Results)")
                bm25_debug_df = results[["clean_text", "bm25_raw"]].copy()
                bm25_debug_df = bm25_debug_df.set_index(
                    bm25_debug_df["clean_text"].str.slice(0, 40) + "..."
                )[["bm25_raw"]]
                st.bar_chart(bm25_debug_df)

    # ===================== TAB 2: DATASET ANALYTICS =====================
    with tab_dataset:
        st.subheader(f"Dataset Analytics (All {len(df)} Threat Descriptions)")

        col_a, col_b = st.columns(2)

        with col_a:
            st.write("### Threat Categories")
            st.bar_chart(df["category"].value_counts())

            st.write("### Attack Vectors")
            st.bar_chart(df["vector"].value_counts())

        with col_b:
            st.write("### Threat Actors")
            st.bar_chart(df["actor"].value_counts().head(10))

            st.write("### Severity Distribution")
            st.bar_chart(df["severity"].value_counts().sort_index())

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


if __name__ == "__main__":
    main()
