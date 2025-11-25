import pandas as pd
import os

# Path from src/ folder
CSV_PATH = os.path.join("..", "data", "raw", "Cybersecurity_Dataset.csv")

def prepare_dataset():

    print(f"[+] Loading dataset from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    print("[+] Columns detected:")
    print(list(df.columns))

    # Rename into unified schema for IR
    df = df.rename(columns={
        "Cleaned Threat Description": "text",
        "Threat Category": "category",
        "Threat Actor": "actor",
        "Attack Vector": "vector",
        "Geographical Location": "location",
        "Severity Score": "severity"
    })

    # Keep only necessary columns
    df_ir = df[["text", "category", "actor", "vector", "location", "severity"]]

    # Make sure text is clean (remove NaN, short, or invalid entries)
    df_ir["text"] = df_ir["text"].astype(str)
    df_ir = df_ir[df_ir["text"].str.len() > 30]  # Keep meaningful documents

    print(f"[+] Filtered dataset size: {len(df_ir)} documents")

    # Save the processed dataset
    out_path = os.path.join("..", "data", "processed", "ir_ready_dataset.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df_ir.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[+] Saved IR-ready dataset to:\n    {out_path}")

    return df_ir


if __name__ == "__main__":
    df = prepare_dataset()
    print(df.head())
