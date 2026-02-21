# data_preprocessing.py

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------
# Paths
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "house_price.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed"

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# Load Data
# ---------------------------------------

df = pd.read_csv(RAW_PATH)

print("Original Shape:", df.shape)

# ---------------------------------------
# Basic Cleaning
# ---------------------------------------

# Drop unnecessary columns
if "id" in df.columns:
    df = df.drop(columns=["id"])

cols_to_drop = ["Date", "Postal Code", "Lattitude", "Longitude"]
df = df.drop(columns=cols_to_drop, errors="ignore")

# Remove duplicates and missing values
df = df.drop_duplicates()
df = df.dropna()

print("After Cleaning:", df.shape)

# ---------------------------------------
# Log Transform Target
# ---------------------------------------

df["Price"] = np.log1p(df["Price"])

# ---------------------------------------
# Save Processed Dataset
# ---------------------------------------

df.to_csv(PROCESSED_PATH / "house_price.csv", index=False)

print("Data preprocessing completed successfully.")
print("Saved at:", PROCESSED_PATH / "house_price.csv")