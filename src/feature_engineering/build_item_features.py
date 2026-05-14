"""
build_item_features.py
======================
Production script to generate ITEM FEATURES for the
Context-Aware Neural Recommendation Engine.

Run from PROJECT ROOT:
    python src/feature_engineering/build_item_features.py

WHY PATH HANDLING MATTERS:
    Python resolves relative paths from the WORKING DIRECTORY — the folder
    you are in when you type the command, NOT the folder this file lives in.
    Running from the project root means all paths must be written relative
    to that root.

    WRONG : ../data/processed/   (tries to go UP from src/feature_engineering/)
    RIGHT : data/processed/      (correct relative to project root)
"""

import os
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# PATH CONFIGURATION
# ──────────────────────────────────────────────
PROCESSED_DIR = "data/processed"

INPUT_ARTICLES     = os.path.join(PROCESSED_DIR, "articles_cleaned.csv")
INPUT_TRANSACTIONS = os.path.join(PROCESSED_DIR, "transactions_cleaned.csv")
OUTPUT_ITEM_FEATURES = os.path.join(PROCESSED_DIR, "item_features.csv")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned articles and transactions datasets."""
    print("📂 Loading cleaned datasets...")

    articles     = pd.read_csv(INPUT_ARTICLES)
    transactions = pd.read_csv(INPUT_TRANSACTIONS, parse_dates=["t_dat"])

    print(f"   ✅ articles      : {articles.shape}")
    print(f"   ✅ transactions  : {transactions.shape}")
    return articles, transactions


def build_popularity_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute item-level popularity signals from transaction history.

    WHY: Popularity is a powerful baseline feature in recommendation systems.
    Frequently bought items are more likely to appeal to new/cold-start users.
    These signals also help the Two-Tower model learn which items are globally
    attractive versus niche.
    """
    print("\n🔧 Building item popularity features...")

    # ── Overall popularity ────────────────────────────────────────────────
    pop = (
        transactions
        .groupby("article_id")
        .agg(
            total_purchases  = ("customer_id", "count"),        # how often bought
            unique_buyers    = ("customer_id", "nunique"),      # breadth of appeal
            avg_price        = ("price", "mean"),               # price signal
            first_sold_date  = ("t_dat", "min"),
            last_sold_date   = ("t_dat", "max"),
        )
        .reset_index()
    )

    # ── Recency of last sale ───────────────────────────────────────────────
    # WHY: Items that haven't sold recently may be out of stock or out of
    # fashion. This feature lets the model down-rank stale inventory.
    reference_date = transactions["t_dat"].max()
    pop["days_since_last_sale"] = (
        reference_date - pop["last_sold_date"]
    ).dt.days

    # ── Normalised popularity score (0–1) ─────────────────────────────────
    # WHY: Raw counts vary hugely; normalising makes the feature usable
    # across different model architectures without re-scaling each time.
    max_purchases = pop["total_purchases"].max()
    pop["popularity_score"] = pop["total_purchases"] / max_purchases

    print(f"   ✅ popularity features shape : {pop.shape}")
    return pop


def build_recent_popularity(
    transactions: pd.DataFrame,
    days: int = 30,
) -> pd.DataFrame:
    """
    Compute item popularity over the most recent N days.

    WHY: Fashion trends change quickly. An item that was popular 6 months ago
    may be completely irrelevant now. Recency-weighted popularity lets the
    model surface trending items rather than just historically popular ones.
    """
    print(f"\n🔧 Building recent popularity (last {days} days)...")

    cutoff = transactions["t_dat"].max() - pd.Timedelta(days=days)
    recent = transactions[transactions["t_dat"] >= cutoff]

    recent_pop = (
        recent
        .groupby("article_id")
        .agg(recent_purchases = ("customer_id", "count"))
        .reset_index()
    )
    recent_pop.rename(
        columns={"recent_purchases": f"purchases_last_{days}d"},
        inplace=True,
    )

    print(f"   ✅ recent popularity shape : {recent_pop.shape}")
    return recent_pop


def merge_item_features(
    articles: pd.DataFrame,
    popularity: pd.DataFrame,
    recent_popularity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge article metadata with computed popularity signals.

    WHY: The item tower of the Two-Tower model needs BOTH static catalogue
    attributes (category, colour, garment group) AND dynamic signals
    (how popular the item is right now). One merged table is cleaner and
    easier to feed into training pipelines.
    """
    print("\n🔗 Merging article metadata with popularity features...")

    item_features = articles.merge(popularity,        on="article_id", how="left")
    item_features = item_features.merge(recent_popularity, on="article_id", how="left")

    # Fill NaNs for articles with zero transactions (never-sold items)
    fill_zero_cols = [
        "total_purchases", "unique_buyers", "popularity_score",
        "days_since_last_sale", "purchases_last_30d",
    ]
    for col in fill_zero_cols:
        if col in item_features.columns:
            item_features[col] = item_features[col].fillna(0)

    print(f"   ✅ item_features shape : {item_features.shape}")
    return item_features


def save_features(item_features: pd.DataFrame) -> None:
    """Save the final item feature table to the processed data directory."""
    print(f"\n💾 Saving item features → {OUTPUT_ITEM_FEATURES}")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    item_features.to_csv(OUTPUT_ITEM_FEATURES, index=False)
    print("   ✅ Saved successfully.")


def main() -> None:
    print("=" * 55)
    print("  ITEM FEATURE ENGINEERING PIPELINE")
    print("=" * 55)

    articles, transactions = load_data()
    popularity        = build_popularity_features(transactions)
    recent_popularity = build_recent_popularity(transactions, days=30)
    item_features     = merge_item_features(articles, popularity, recent_popularity)
    save_features(item_features)

    print("\n🎯 Item feature engineering complete.")
    print(f"   Output : {OUTPUT_ITEM_FEATURES}")
    print(f"   Shape  : {item_features.shape}")
    print("=" * 55)


if __name__ == "__main__":
    main()
