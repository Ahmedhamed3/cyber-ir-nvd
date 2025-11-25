import os
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
import joblib  # to save models/vectorizers


# ---------- PATHS ----------
# This should point to your FINAL dataset that already has `clean_text`.
# Put your new CSV there and name it exactly like this:
RAW_CSV = os.path.join("..", "data", "processed", "ir_preprocessed_dataset.csv")

BOW_PATH = os.path.join("..", "data", "processed", "bow_matrix.npz")
TFIDF_PATH = os.path.join("..", "data", "processed", "tfidf_matrix.npz")
BOW_VECTORIZER_PATH = os.path.join("..", "models", "bow_vectorizer.pkl")
TFIDF_VECTORIZER_PATH = os.path.join("..", "models", "tfidf_vectorizer.pkl")
# ----------------------------


def main():
    print(f"[+] Loading FINAL dataset from: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)

    # We assume clean_text is already prepared and should NOT be changed
    if "clean_text" not in df.columns:
        raise ValueError("Expected a 'clean_text' column in the dataset.")

    # Make sure it's string, don't change content
    df["clean_text"] = df["clean_text"].astype(str)
    print(f"[+] Number of documents: {len(df)}")

    # --------- VECTORIZATION ---------
    texts = df["clean_text"].tolist()

    print("[+] Building Bag-of-Words (BoW) model on clean_text...")
    bow_vectorizer = CountVectorizer(
        max_df=0.95,   # ignore extremely common terms
        min_df=1,      # keep all terms that appear at least once
    )
    X_bow = bow_vectorizer.fit_transform(texts)
    print(f"    BoW matrix shape: {X_bow.shape}")

    print("[+] Building TF-IDF model on clean_text...")
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=1,
    )
    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    print(f"    TF-IDF matrix shape: {X_tfidf.shape}")

    # Save sparse matrices (these are what SearchEngine uses)
    os.makedirs(os.path.dirname(BOW_PATH), exist_ok=True)
    sparse.save_npz(BOW_PATH, X_bow)
    sparse.save_npz(TFIDF_PATH, X_tfidf)
    print(f"[+] Saved BoW matrix to: {BOW_PATH}")
    print(f"[+] Saved TF-IDF matrix to: {TFIDF_PATH}")

    # Save vectorizers
    os.makedirs(os.path.dirname(BOW_VECTORIZER_PATH), exist_ok=True)
    joblib.dump(bow_vectorizer, BOW_VECTORIZER_PATH)
    joblib.dump(tfidf_vectorizer, TFIDF_VECTORIZER_PATH)
    print(f"[+] Saved BoW vectorizer to: {BOW_VECTORIZER_PATH}")
    print(f"[+] Saved TF-IDF vectorizer to: {TFIDF_VECTORIZER_PATH}")

    print("[+] DONE. Dataset itself was NOT modified.")


if __name__ == "__main__":
    main()
