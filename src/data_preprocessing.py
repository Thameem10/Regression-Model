import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------
# Paths
# ---------------------------------------

BASE_DIR       = Path(__file__).resolve().parent.parent
RAW_PATH       = BASE_DIR / "data" / "raw" / "house_price.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed"
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# Load Data
# ---------------------------------------

df = pd.read_csv(RAW_PATH)
print("Original Shape:", df.shape)

# ---------------------------------------
# Step 1: Drop Unnecessary Columns
# ---------------------------------------

drop_cols = [
    "id", "Date", "Postal Code", "Longitude",
    "Renovation Year", "number of views", "waterfront present" , 
    "lot area" , "lot_area_renov" , "Built Year" , "condition of the house" , 
    "Number of schools nearby" , "Distance from the airport"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
print("After dropping noise columns:", df.shape)

# ---------------------------------------
# Step 2: Remove Duplicates
# ---------------------------------------

before = len(df)
df = df.drop_duplicates()
print(f"Duplicates removed: {before - len(df)} rows dropped.")

# ---------------------------------------
# Step 3: Fill Missing Values with Mean
# (instead of dropping rows)
# ---------------------------------------

missing_before = df.isnull().sum()
missing_cols   = missing_before[missing_before > 0]

if not missing_cols.empty:
    print(f"\nMissing values found:")
    print(missing_cols)
    for col in missing_cols.index:
        df[col] = df[col].fillna(df[col].mean())
    print("✔ Missing values filled with column mean.")
else:
    print("✔ No missing values found.")

print("After handling missing values:", df.shape)

# ---------------------------------------
# Step 4: Drop Near-Zero Variance Columns
# (columns where almost all values are the same
#  — these carry no useful information)
# ---------------------------------------

num_df    = df.select_dtypes(include=[np.number])
variance  = num_df.var()
low_var   = variance[variance < 0.01].index.tolist()

# Never drop the target column
if "Price" in low_var:
    low_var.remove("Price")

if low_var:
    print(f"\n⚠ Near-zero variance columns dropped: {low_var}")
    df = df.drop(columns=low_var)
else:
    print("✔ No near-zero variance columns found.")

print("After variance filter:", df.shape)

# ---------------------------------------
# Step 5: Log Transform Target (Price)
# ---------------------------------------

df["Price"] = np.log1p(df["Price"])
print("✔ Log transform applied to Price.")

# ---------------------------------------
# Step 6: Min-Max Scaling
# Scales every feature to range [0, 1]
# Good for models sensitive to feature magnitude
# ---------------------------------------

# Separate target before scaling
target     = df["Price"].copy()
features   = df.drop(columns=["Price"])

scaler        = MinMaxScaler()
scaled_array  = scaler.fit_transform(features)
df_scaled     = pd.DataFrame(scaled_array, columns=features.columns)

# Put target back (not scaled — already log-transformed)
df_scaled["Price"] = target.values

print(f"✔ Min-Max Scaling applied to {len(features.columns)} features.")
print(f"   All feature values now in range [0, 1]")



# ---------------------------------------
# Step 7: Correlation Heatmap
# This helps you:
#  1. See which features are related to Price (target)
#  2. Spot redundant features (corr > 0.8 with each other)
#  3. Remove features with near-zero correlation to everything
# ---------------------------------------

print("\nGenerating correlation heatmap...")

corr_matrix = df_scaled.corr()

# --- Full Heatmap ---
plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    annot=True,           # show numbers inside cells
    fmt=".2f",            # 2 decimal places
    cmap="coolwarm",      # red=positive, blue=negative
    center=0,
    linewidths=0.5,
    linecolor="white",
    annot_kws={"size": 7}
)
plt.title("Feature Correlation Heatmap (Min-Max Scaled)", fontsize=13)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(PROCESSED_PATH / "correlation_heatmap.png", dpi=150)
plt.close()
print("✔ Full heatmap saved.")

# --- Price Correlation Bar Chart ---
# Shows which features correlate most with Price
# Easier to read than full heatmap for feature selection

price_corr = corr_matrix["Price"].drop("Price").sort_values(ascending=False)

plt.figure(figsize=(10, 6))
colors = ["tomato" if v < 0 else "steelblue" for v in price_corr.values]
plt.barh(price_corr.index[::-1], price_corr.values[::-1], color=colors[::-1])
plt.axvline(x=0.3,  color="green",  linestyle="--", alpha=0.7, label="Threshold +0.3")
plt.axvline(x=-0.3, color="orange", linestyle="--", alpha=0.7, label="Threshold -0.3")
plt.title("Feature Correlation with Price (Target)", fontsize=12)
plt.xlabel("Correlation Coefficient")
plt.legend()
plt.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(PROCESSED_PATH / "price_correlation_bar.png", dpi=150)
plt.close()
print("✔ Price correlation bar chart saved.")

# --- Print Correlation Summary ---
print("\n" + "="*50)
print("  CORRELATION WITH PRICE (sorted)")
print("="*50)
print(price_corr.round(3).to_string())

print("\n" + "="*50)
print("  FEATURE SELECTION GUIDE")
print("="*50)

strong    = price_corr[abs(price_corr) >= 0.3].index.tolist()
moderate  = price_corr[(abs(price_corr) >= 0.1) & (abs(price_corr) < 0.3)].index.tolist()
weak      = price_corr[abs(price_corr) < 0.1].index.tolist()

print(f"\n✔ Strong correlation with Price (|r| ≥ 0.3) → KEEP:")
for f in strong:
    print(f"   {f:45s} : {price_corr[f]:.3f}")

print(f"\n⚠ Moderate correlation (0.1 ≤ |r| < 0.3) → OPTIONAL:")
for f in moderate:
    print(f"   {f:45s} : {price_corr[f]:.3f}")

print(f"\n✘ Weak correlation with Price (|r| < 0.1) → CONSIDER DROPPING:")
for f in weak:
    print(f"   {f:45s} : {price_corr[f]:.3f}")

# ---------------------------------------
# Step 8: Save Processed Dataset
# ---------------------------------------

df_scaled.to_csv(PROCESSED_PATH / "house_price.csv", index=False)

print("\n" + "="*50)
print(f"✔ Processed dataset saved.")
print(f"   Shape  : {df_scaled.shape}")
print(f"   Path   : {PROCESSED_PATH / 'house_price.csv'}")
print("="*50)