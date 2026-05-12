import pandas as pd


def load_data(file_path):

    df = pd.read_csv(file_path)

    print("Dataset Loaded Successfully")

    return df


def check_missing_values(df):

    print(df.isnull().sum())


def remove_duplicates(df):

    df = df.drop_duplicates()

    return df