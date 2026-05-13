import pandas as pd


# -------------------------------
# Load Dataset
# -------------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully!\n")
    return df


# -------------------------------
# Understand Dataset
# -------------------------------
def understand_data(df):
    print("First 5 Rows:\n")
    print(df.head())

    print("\nDataset Info:\n")
    print(df.info())

    print("\nColumn Names:\n")
    print(df.columns)

    print("\nDataset Shape:")
    print(df.shape)


# -------------------------------
# Check Missing Values
# -------------------------------
def check_missing_values(df):
    print("\nMissing Values:\n")
    print(df.isnull().sum())


# -------------------------------
# Handle Missing Values
# -------------------------------
def handle_missing_values(df):

    # Fill missing descriptions
    if "detail_desc" in df.columns:
        df["detail_desc"] = df["detail_desc"].fillna("No Description")

    return df


# -------------------------------
# Remove Duplicates
# -------------------------------
def remove_duplicates(df):

    print("\nDuplicate Rows:", df.duplicated().sum())

    df = df.drop_duplicates()

    return df


# -------------------------------
# Convert Datatypes
# -------------------------------
def convert_datatypes(df):

    # Convert article_id to string
    if "article_id" in df.columns:
        df["article_id"] = df["article_id"].astype(str)

    return df


# -------------------------------
# Save Cleaned Dataset
# -------------------------------
def save_data(df, output_path):

    df.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved to:\n{output_path}")


# -------------------------------
# Main Function
# -------------------------------
def main():

    input_path = "data/raw/articles.csv/articles.csv"

    output_path = "data/processed/articles_cleaned.csv"

    # Load
    df = load_data(input_path)

    # Understand
    understand_data(df)

    # Missing values
    check_missing_values(df)

    # Clean data
    df = handle_missing_values(df)

    # Remove duplicates
    df = remove_duplicates(df)

    # Convert datatypes
    df = convert_datatypes(df)

    # Save cleaned data
    save_data(df, output_path)

    print("\nPreprocessing Completed Successfully!")


# Run Main
if __name__ == "__main__":
    main()