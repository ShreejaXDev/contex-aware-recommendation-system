"""
build_user_features.py
======================
Production script to generate USER FEATURES for the
Context-Aware Neural Recommendation Engine.

Run from PROJECT ROOT:
    python src/feature_engineering/build_user_features.py

WHY PATH HANDLING MATTERS:
    Python resolves relative paths from wherever you RUN the script, not
    where the script FILE lives. Since we always run from the project root
    (e.g. D:/context-aware-recommendation-system/), all paths must be
    written relative to that root — NOT relative to this script's location.

    WRONG : ../data/processed/   (goes UP from src/feature_engineering/)
    RIGHT : data/processed/      (stays inside the project root)
"""

import os
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# PATH CONFIGURATION
# ──────────────────────────────────────────────
# All paths are relative to the PROJECT ROOT.
# Scripts are always executed from the project root, so this is safe
# and consistent across Windows, macOS, and Linux.

PROCESSED_DIR = "data/processed"

INPUT_CUSTOMERS   = os.path.join(PROCESSED_DIR, "customers_cleaned.csv")
INPUT_TRANSACTIONS = os.path.join(PROCESSED_DIR, "transactions_cleaned.csv")
OUTPUT_USER_FEATURES = os.path.join(PROCESSED_DIR, "user_features.csv")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned customers and transactions datasets."""
    print("📂 Loading cleaned datasets...")

    customers    = pd.read_csv(INPUT_CUSTOMERS)
    transactions = pd.read_csv(INPUT_TRANSACTIONS, parse_dates=["t_dat"])

    print(f"   ✅ customers    : {customers.shape}")
    print(f"   ✅ transactions : {transactions.shape}")
    return customers, transactions


def build_purchase_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data into per-user features.

    WHY: Recommendation models need a compact representation of each user's
    historical behaviour. Raw transaction rows can't be fed directly into a
    model — we summarise them into meaningful signals instead.
    """
    print("\n🔧 Building purchase features...")

    agg = (
        transactions
        .groupby("customer_id")
        .agg(
            purchase_count    = ("article_id", "count"),        # total items bought
            recent_purchase_date = ("t_dat", "max"),            # last purchase date
            first_purchase_date  = ("t_dat", "min"),            # first purchase date
            total_spend       = ("price", "sum"),               # lifetime spend
            avg_spend_per_txn = ("price", "mean"),              # basket size signal
            unique_articles   = ("article_id", "nunique"),      # breadth of taste
        )
        .reset_index()
    )

    # ── Recency: days since last purchase ──────────────────────────────────
    # WHY: Recent purchasers are more likely to engage with recommendations.
    # This is the "R" in classic RFM (Recency-Frequency-Monetary) analysis.
    reference_date = transactions["t_dat"].max()
    agg["days_since_last_purchase"] = (
        reference_date - agg["recent_purchase_date"]
    ).dt.days

    # ── Frequency: purchases per active day ────────────────────────────────
    # WHY: A user who bought 20 items in 20 days is very different from one
    # who bought 20 items over 2 years. Frequency normalises for tenure.
    active_days = (
        agg["recent_purchase_date"] - agg["first_purchase_date"]
    ).dt.days.clip(lower=1)                                     # avoid division by zero
    agg["purchase_frequency"] = agg["purchase_count"] / active_days

    # ── Active status: purchased in the last 90 days ───────────────────────
    # WHY: Binary flag that lets downstream models quickly separate active
    # from lapsed users — useful for candidate generation filtering.
    agg["active_status"] = (agg["days_since_last_purchase"] <= 90).astype(int)

    print(f"   ✅ purchase features shape : {agg.shape}")
    return agg


def merge_with_customers(
    customers: pd.DataFrame,
    purchase_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge demographic customer attributes with computed purchase features.

    WHY: The Two-Tower model's user tower needs BOTH static demographics
    (age, club membership) AND dynamic behaviour signals (how recently/
    frequently the user shops). Merging produces a single feature table.
    """
    print("\n🔗 Merging customer demographics with purchase features...")

    user_features = customers.merge(purchase_features, on="customer_id", how="left")

    # Fill NaNs for customers with no transaction history
    # (cold-start users — important edge case in recommendation systems)
    user_features["purchase_count"]          = user_features["purchase_count"].fillna(0)
    user_features["days_since_last_purchase"] = user_features["days_since_last_purchase"].fillna(9999)
    user_features["active_status"]            = user_features["active_status"].fillna(0).astype(int)
    user_features["purchase_frequency"]       = user_features["purchase_frequency"].fillna(0.0)
    user_features["total_spend"]              = user_features["total_spend"].fillna(0.0)

    print(f"   ✅ user_features shape : {user_features.shape}")
    return user_features


def save_features(user_features: pd.DataFrame) -> None:
    """Save the final user feature table to the processed data directory."""
    print(f"\n💾 Saving user features → {OUTPUT_USER_FEATURES}")
    os.makedirs(PROCESSED_DIR, exist_ok=True)   # create folder if missing
    user_features.to_csv(OUTPUT_USER_FEATURES, index=False)
    print("   ✅ Saved successfully.")


def main() -> None:
    print("=" * 55)
    print("  USER FEATURE ENGINEERING PIPELINE")
    print("=" * 55)

    customers, transactions = load_data()
    purchase_features = build_purchase_features(transactions)
    user_features     = merge_with_customers(customers, purchase_features)
    save_features(user_features)

    print("\n🎯 User feature engineering complete.")
    print(f"   Output : {OUTPUT_USER_FEATURES}")
    print(f"   Shape  : {user_features.shape}")
    print("=" * 55)


if __name__ == "__main__":
    main()
