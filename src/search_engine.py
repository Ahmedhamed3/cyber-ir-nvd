import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# NLTK for query preprocessing (same style as your corpus)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class SearchEngine:
    """
    Cybersecurity Threat Search Engine

    Uses:
      - Overlap-based scoring for TF-IDF search name:
          score = (# query tokens found in doc) / (# query tokens)
      - BM25
      - Simple Boolean search over clean_text

    Supports filters on: category, actor, vector, location, severity.
    """

    def __init__(self):
        # --- Paths (relative to src/) ---
        base_data = os.path.join("..", "data", "processed")
        base_models = os.path.join("..", "models")

        csv_path = os.path.join(base_data, "ir_preprocessed_dataset.csv")
        tfidf_path = os.path.join(base_data, "tfidf_matrix.npz")
        tfidf_vec_path = os.path.join(base_models, "tfidf_vectorizer.pkl")

        # Load dataset
        print(f"[+] Loading preprocessed dataset from: {csv_path}")
        self.df = pd.read_csv(csv_path)

        if "clean_text" not in self.df.columns:
            raise ValueError("Expected a 'clean_text' column in ir_preprocessed_dataset.csv")

        # Load TF-IDF matrix and vectorizer (still loaded; useful for similarity, etc.)
        print(f"[+] Loading TF-IDF matrix from: {tfidf_path}")
        self.X_tfidf = sparse.load_npz(tfidf_path)

        print(f"[+] Loading TF-IDF vectorizer from: {tfidf_vec_path}")
        self.tfidf_vectorizer = joblib.load(tfidf_vec_path)

        # Prepare BM25 corpus (list of tokens per document)
        print("[+] Preparing BM25 corpus...")
        self.tokenized_docs = [doc.split() for doc in self.df["clean_text"].astype(str)]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print("[+] BM25 index built.")

        # Precompute token sets for overlap scoring
        self.doc_token_sets = [
            set(str(doc).split()) for doc in self.df["clean_text"].astype(str)
        ]

        # Build preprocessing tools for queries (same style as corpus)
        self._build_query_preprocessor()

        # Cache: number of documents
        self.n_docs = len(self.df)
        print(f"[+] SearchEngine ready with {self.n_docs} documents.\n")

    # ------------------------------------------------------------------
    # Preprocessing for queries
    # ------------------------------------------------------------------
    def _build_query_preprocessor(self):
        # one-time downloads (quiet)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def _preprocess_query(self, query: str) -> str:
        """
        Apply similar preprocessing to the query as done for clean_text:
        lowercase, remove non-letters, tokenize, remove stopwords, lemmatize.
        Returns a cleaned string.
        """
        text = query.lower()
        # remove URLs
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        # keep only letters and spaces
        text = re.sub(r"[^a-z\s]", " ", text)

        tokens = word_tokenize(text)
        # remove stopwords + very short tokens
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        # lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    def _apply_filters(self, df_results: pd.DataFrame, filters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Apply filters like:
          filters = {
            "category": "DDoS",
            "actor": "APT-28",
            "vector": "Email",
            "location": "North Korea",
            "severity": 5
          }
        Any key can be omitted.
        """
        if not filters:
            return df_results

        mask = pd.Series(True, index=df_results.index)

        for key, value in filters.items():
            if value is None or value == "" or (isinstance(value, str) and value.lower() == "all"):
                continue
            if key not in df_results.columns:
                continue
            mask &= (df_results[key] == value)

        return df_results[mask]

    # ------------------------------------------------------------------
    # TF-IDF / VSM search (but score = token overlap)
    # ------------------------------------------------------------------
    def search_tfidf(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        OVERLAP-BASED SEARCH

        Score is based purely on how many query tokens appear in the document:

            score = (# matched query tokens) / (# query tokens)

        So:
          - score = 1.0 -> all query words are present in clean_text
          - score = 0.5 -> half of the query words appear
          - score = 0.0 -> none of the query words appear
        """
        if not query.strip():
            raise ValueError("Query is empty.")

        clean_query = self._preprocess_query(query)
        if not clean_query:
            raise ValueError("Query became empty after preprocessing.")

        query_tokens = clean_query.split()
        if not query_tokens:
            raise ValueError("Query has no tokens after preprocessing.")

        query_set = set(query_tokens)
        q_len = len(query_set)

        # Compute overlap score for each document
        scores = []
        for doc_tokens in self.doc_token_sets:
            matched = len(query_set.intersection(doc_tokens))
            score = matched / q_len  # fraction of query tokens found
            scores.append(score)

        scores = np.array(scores, dtype=float)

         # Build DataFrame with scores, apply filters, then drop zero/near-zero results
        df_results = self.df.copy()
        df_results["score"] = scores
        df_results = self._apply_filters(df_results, filters)

       # Remove documents with zero (or negligible) score before ranking/limiting
        epsilon = 1e-8
        df_results = df_results[df_results["score"] > epsilon]

        if df_results.empty:
            return df_results

        # Rank remaining documents and then cap at top_k
        if "severity" in df_results.columns:
            df_results = df_results.sort_values(
                by=["score", "severity"],
                ascending=[False, False]
            )
        else:
            df_results = df_results.sort_values("score", ascending=False)

        return df_results.head(top_k)

    # ------------------------------------------------------------------
    # BM25 search
    # ------------------------------------------------------------------
    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Search using BM25.
        - bm25_raw: original BM25 score
        - score: normalized bm25_raw in [0, 1] for display
        Ranking is done by bm25_raw.
        """
        if not query.strip():
            raise ValueError("Query is empty.")

        clean_query = self._preprocess_query(query)
        tokens = clean_query.split()
        if not tokens:
            raise ValueError("Query became empty after preprocessing.")

        # Raw BM25 scores for all docs
        scores = np.array(self.bm25.get_scores(tokens), dtype=float)

        # Build DataFrame with raw scores, apply filters, then drop zero/near-zero results
        df_results = self.df.copy()
        df_results["bm25_raw"] = scores
        df_results = self._apply_filters(df_results, filters)
        epsilon = 1e-8
        df_results = df_results[df_results["bm25_raw"] > epsilon]

        if df_results.empty:
            return df_results

         # Normalize bm25_raw to [0, 1] for display (after filtering so max>0)
        max_raw = df_results["bm25_raw"].max()
        df_results["score"] = df_results["bm25_raw"] / max_raw

        # Optional tie-break using severity, then cap at top_k
        if "severity" in df_results.columns:
            df_results = df_results.sort_values(
                by=["score", "bm25_raw", "severity"],
                ascending=[False, False, False]
            )
        else:
            df_results = df_results.sort_values(
                by=["score", "bm25_raw"],
                ascending=[False, False]
            )

        return df_results.head(top_k)

    # ------------------------------------------------------------------
    # Boolean search (simple AND model)
    # Boolean search with AND / OR / NOT and parentheses
    # ------------------------------------------------------------------
    def search_boolean(
        self,
        query: str,
        filters: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Boolean model supporting AND / OR / NOT with optional parentheses.

        Rules:
        - Operators are AND, OR, NOT (case-insensitive).
        - If no operator is specified between terms, AND is assumed.
        - Terms are matched against `clean_text`.
        - Matching documents receive score = 1.0.
        """

        def _tokenize_boolean_query(raw_query: str) -> List[str]:
            pattern = r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|\w+"
            raw_tokens = re.findall(pattern, raw_query, flags=re.IGNORECASE)

            processed_tokens: List[str] = []
            for tok in raw_tokens:
                upper_tok = tok.upper()
                if upper_tok in {"AND", "OR", "NOT", "(", ")"}:
                    processed_tokens.append(upper_tok)
                    continue

                clean_tok = self._preprocess_query(tok).strip()
                if not clean_tok:
                    continue
                # _preprocess_query may expand into multiple tokens (rare); keep each
                processed_tokens.extend(clean_tok.split())

            return processed_tokens

        def _insert_implicit_and(tokens: List[str]) -> List[str]:
            if not tokens:
                return tokens

            result: List[str] = []
            for i, tok in enumerate(tokens):
                if i > 0:
                    prev = result[-1]
                    if (
                        (prev not in {"AND", "OR", "NOT", "("})
                        or prev == ")"
                    ):
                        if tok in {"(", "NOT"} or tok not in {"AND", "OR", ")"}:
                            result.append("AND")
                result.append(tok)
            return result

        def _to_postfix(tokens: List[str]) -> List[str]:
            precedence = {"NOT": 3, "AND": 2, "OR": 1}
            output: List[str] = []
            stack: List[str] = []

            for tok in tokens:
                if tok in {"AND", "OR", "NOT"}:
                    if tok == "NOT":
                        while stack and precedence.get(stack[-1], 0) > precedence[tok]:
                            output.append(stack.pop())
                    else:
                        while stack and stack[-1] != "(" and precedence.get(stack[-1], 0) >= precedence[tok]:
                            output.append(stack.pop())
                    stack.append(tok)
                elif tok == "(":
                    stack.append(tok)
                elif tok == ")":
                    while stack and stack[-1] != "(":
                        output.append(stack.pop())
                    if not stack:
                        raise ValueError("Mismatched parentheses in Boolean query.")
                    stack.pop()  # Remove '('
                else:
                    output.append(tok)

            while stack:
                if stack[-1] in {"(", ")"}:
                    raise ValueError("Mismatched parentheses in Boolean query.")
                output.append(stack.pop())

            return output

        def _evaluate_postfix(postfix: List[str], doc_tokens: set[str]) -> bool:
            stack: List[bool] = []
            for tok in postfix:
                if tok == "NOT":
                    if not stack:
                        raise ValueError("Invalid Boolean expression: NOT without operand.")
                    operand = stack.pop()
                    stack.append(not operand)
                elif tok in {"AND", "OR"}:
                    if len(stack) < 2:
                        raise ValueError("Invalid Boolean expression: operator without operands.")
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a and b if tok == "AND" else a or b)
                else:
                    stack.append(tok.lower() in doc_tokens)

            if len(stack) != 1:
                raise ValueError("Invalid Boolean expression: unresolved terms remain.")

            return stack[0]

        if not query.strip():
            raise ValueError("Query is empty.")

        tokens_raw = _tokenize_boolean_query(query)
        if not tokens_raw:
            raise ValueError("Query became empty after preprocessing.")

        tokens_with_and = _insert_implicit_and(tokens_raw)
        postfix_expr = _to_postfix(tokens_with_and)

        matches = []
        for idx, row in self.df.iterrows():
            clean_text = str(row.get("clean_text", ""))
            doc_tokens = set(clean_text.lower().split())

            try:
                if _evaluate_postfix(postfix_expr, doc_tokens):
                    matches.append(idx)
            except ValueError as e:
                raise ValueError("Invalid Boolean query. Please use AND/OR/NOT and parentheses.") from e

        df_results = self.df.loc[matches].copy()

        if df_results.empty:
            return df_results

        df_results["score"] = 1.0  # Boolean = match / no match

        df_results = self._apply_filters(df_results, filters)

       
        if "severity" in df_results.columns:
            df_results = df_results.sort_values(by="severity", ascending=False)

        return df_results

    # ------------------------------------------------------------------
    # Similar documents
    # ------------------------------------------------------------------
    def get_similar(
        self,
        doc_index: int,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """
        Given a document index (row index in df),
        return the most similar documents based on TF-IDF cosine similarity.
        (This still uses the original TF-IDF matrix.)
        """
        if doc_index < 0 or doc_index >= self.n_docs:
            raise IndexError("doc_index out of range")

        doc_vec = self.X_tfidf[doc_index]
        sims = cosine_similarity(doc_vec, self.X_tfidf)[0]

        top_idx = np.argsort(sims)[::-1]
        # Exclude the document itself
        top_idx = top_idx[top_idx != doc_index]

        df_results = self.df.iloc[top_idx].copy()
        df_results["score"] = sims[top_idx]

        return df_results.head(top_k)
