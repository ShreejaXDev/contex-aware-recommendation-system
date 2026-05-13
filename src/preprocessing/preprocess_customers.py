"""
preprocess_customers.py
=======================
Production preprocessing script for the H&M customers dataset.

Purpose:
    - Load raw customers.csv
    - Clean and standardize all columns
    - Handle missing values using domain-appropriate strategies
    - Remove duplicates
    - Fix data types
    - Save cleaned output to data/processed/customers_cleaned.csv

How this fits into the pipeline:
    RAW DATA → [THIS SCRIPT] → customers_cleaned.csv → Feature Engineering → Model

Author : Context-Aware Neural Recommendation Engine — Internship Project
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# ── Logger Setup ──────────────────────────────────────────────────────────────
# Using Python's logging module instead of print() — this is the production way.
# Logs can be redirected to files, monitoring systems, etc.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ── Path Constants ────────────────────────────────────────────────────────────
# Using os.path so this works on Windows, Mac, and Linux equally
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH  = os.path.join(BASE_DIR, 'data', 'raw',       'customers.csv')
OUTPUT_PATH    = os.path.join(BASE_DIR, 'data', 'processed', 'customers_cleaned.csv')


# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

def load_raw_customers(filepath: str) -> pd.DataFrame:
    """
    Load raw customers CSV from disk.

    Args:
        filepath: Absolute path to customers.csv

    Returns:
        Raw DataFrame as loaded from disk (no modifications yet)

    Why separate function?
        Isolating load logic makes it easy to swap the source later
        (e.g., load from S3, database, Parquet) without touching cleaning logic.
    """
    logger.info(f"Loading customers data from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"customers.csv not found at: {filepath}\n"
            f"Make sure you have downloaded the H&M dataset into data/raw/"
        )

    df = pd.read_csv(filepath)

    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


# =============================================================================
# SECTION 2 — DUPLICATE REMOVAL
# =============================================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate customer records.

    Strategy:
        Each customer should appear exactly once (identified by customer_id).
        If a customer_id appears multiple times, keep the first occurrence.

    Why this matters for recommendations:
        Duplicate customers inflate interaction counts in collaborative filtering,
        making some users appear more active than they really are.

    Args:
        df: Raw customers DataFrame

    Returns:
        DataFrame with duplicate rows removed
    """
    before = len(df)

    # Remove exact full-row duplicates first
    df = df.drop_duplicates()

    # Remove key-based duplicates — keep first occurrence per customer_id
    df = df.drop_duplicates(subset=['customer_id'], keep='first')

    removed = before - len(df)
    if removed > 0:
        logger.info(f"Removed {removed:,} duplicate customer records")
    else:
        logger.info("No duplicates found in customers data")

    return df


# =============================================================================
# SECTION 3 — HANDLE MISSING VALUES
# =============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill or drop missing values using domain-appropriate strategies.

    Strategy per column:
        - customer_id : Drop rows (cannot identify user without this)
        - FN          : Fill with 0 (binary flag — null ≈ not flagged)
        - Active      : Fill with 0 (binary flag — null ≈ inactive)
        - club_member_status    : Fill with 'Unknown'
        - fashion_news_frequency: Fill with 'None'
        - age         : Fill with median (robust to age distribution skew)

    Why median for age?
        The median is not affected by extreme outliers (e.g., age=120).
        Using mean would pull the imputed value toward outliers.

    Args:
        df: DataFrame after duplicate removal

    Returns:
        DataFrame with nulls handled
    """
    logger.info("Handling missing values in customers data...")

    # Drop rows with no customer_id — these cannot be used in any recommendation
    before = len(df)
    df = df.dropna(subset=['customer_id'])
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped:,} rows with null customer_id")

    # ── Binary flag columns — null means 0 (not set) ─────────────────────────
    for col in ['FN', 'Active']:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            df[col] = df[col].fillna(0)
            if null_count > 0:
                logger.info(f"  '{col}': filled {null_count:,} nulls → 0")

    # ── Categorical columns — null becomes a valid 'Unknown' category ─────────
    categorical_fills = {
        'club_member_status':     'Unknown',
        'fashion_news_frequency': 'None',
    }
    for col, fill_value in categorical_fills.items():
        if col in df.columns:
            null_count = df[col].isnull().sum()
            df[col] = df[col].fillna(fill_value)
            if null_count > 0:
                logger.info(f"  '{col}': filled {null_count:,} nulls → '{fill_value}'")

    # ── Age — fill with median, then clip to realistic human age range ─────────
    if 'age' in df.columns:
        null_count = df['age'].isnull().sum()
        age_median = df['age'].median()
        df['age'] = df['age'].fillna(age_median)
        # Clip unrealistic ages — no H&M customer should be 0 or 200
        df['age'] = df['age'].clip(lower=15, upper=100)
        if null_count > 0:
            logger.info(f"  'age': filled {null_count:,} nulls → median ({age_median:.1f})")

    remaining_nulls = df.isnull().sum().sum()
    logger.info(f"Missing value handling complete. Remaining nulls: {remaining_nulls}")

    return df


# =============================================================================
# SECTION 4 — FIX DATA TYPES
# =============================================================================

def fix_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure each column has the correct data type.

    Why types matter:
        Wrong types cause silent bugs — e.g., sorting '10' < '9' as strings
        is incorrect. Integers are faster and use less memory than floats.
        String IDs ensure consistent join behavior across datasets.

    Args:
        df: DataFrame after null handling

    Returns:
        DataFrame with corrected column types
    """
    logger.info("Fixing data types in customers data...")

    # customer_id must be a string for consistent joins with transactions
    if 'customer_id' in df.columns:
        df['customer_id'] = df['customer_id'].astype(str).str.strip()

    # Binary flags — convert to integer (0 or 1)
    for col in ['FN', 'Active']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Age — convert to integer after filling nulls
    if 'age' in df.columns:
        df['age'] = df['age'].astype(int)

    # Categorical columns — strip whitespace and standardize
    cat_cols = ['club_member_status', 'fashion_news_frequency']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    logger.info("Data types fixed successfully")
    return df


# =============================================================================
# SECTION 5 — SAVE OUTPUT
# =============================================================================

def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned DataFrame to the processed/ directory.

    Args:
        df: Fully cleaned customers DataFrame
        output_path: Where to save the CSV file
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    # Verify the file was actually saved correctly
    file_size_mb = os.path.getsize(output_path) / 1024**2
    logger.info(f"Saved cleaned customers → {output_path}")
    logger.info(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}  |  Size: {file_size_mb:.2f} MB")


# =============================================================================
# SECTION 6 — MAIN PIPELINE FUNCTION
# =============================================================================

def preprocess_customers(
    input_path: str  = RAW_DATA_PATH,
    output_path: str = OUTPUT_PATH
) -> pd.DataFrame:
    """
    Main entry point — runs the full customers preprocessing pipeline.

    This function orchestrates all the steps in the correct order.
    It's designed to be callable from other scripts or orchestration tools.

    Pipeline order:
        1. Load raw data
        2. Remove duplicates
        3. Handle missing values
        4. Fix data types
        5. Save cleaned output

    Args:
        input_path:  Path to raw customers.csv
        output_path: Path where cleaned CSV will be saved

    Returns:
        Cleaned customers DataFrame (also saved to disk)
    """
    logger.info("=" * 60)
    logger.info("  CUSTOMERS PREPROCESSING PIPELINE — START")
    logger.info("=" * 60)

    start_time = datetime.now()

    # Execute pipeline steps in order
    df = load_raw_customers(input_path)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = fix_datatypes(df)
    save_cleaned_data(df, output_path)

    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info("=" * 60)
    logger.info(f"  CUSTOMERS PREPROCESSING COMPLETE in {elapsed:.2f}s")
    logger.info("=" * 60)

    return df


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    """
    Run this script directly: python preprocess_customers.py
    
    This block only executes when the script is run directly,
    NOT when it's imported by another module (e.g., a pipeline orchestrator).
    This is a Python best practice for production scripts.
    """
    cleaned_df = preprocess_customers()

    # Quick sanity check — print a preview of the cleaned output
    print("\n--- Cleaned Customers Preview (first 5 rows) ---")
    print(cleaned_df.head())
    print(f"\nFinal shape: {cleaned_df.shape}")
    print(f"Null count : {cleaned_df.isnull().sum().sum()}")
