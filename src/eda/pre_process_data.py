import pandas as pd
import os

# Base project _______________________________________________________________________________________________

# This function preprocesses the reviews from the raw dataset
def preprocess_reviews(
    columns: list = []
) -> None:

    if not os.path.exists("data/_preprocessed/reviews.csv") or len(columns) > 0:
        os.makedirs("data/_preprocessed/") if not os.path.exists(
            "data/_preprocessed/"
        ) else None
        df = pd.read_json("data/_raw/review_Software.jsonl", lines=True)
        
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Le seguenti colonne non esistono nel DataFrame: {missing_columns}")
        
        # Drop duplicates and raws with missing values
        df.drop_duplicates(subset=columns, inplace=True)
        df.dropna(subset=columns, inplace=True)
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime(
            "%Y-%m-%d"
        )
        # Sort by timestamp
        df = df.sort_values(by="timestamp")
        
        # Select only the specified columns
        df = df[columns]
        
        df.to_csv("data/_preprocessed/reviews.csv", index=False)


# This function filters the reviews by a given date range
def filter_by_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    df_filtered = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
    return df_filtered

# This function filters the reviews by the number of reviews per user
def filter_by_number_reviews_user(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df_grouped = df[["user_id", "rating"]].groupby("user_id").count().reset_index()
    df_grouped.columns = ["user_id", "count"]
    df_filtered = df[ df["user_id"].isin(df_grouped[df_grouped["count"] >= n]["user_id"]) ]
    return df_filtered

# This function filters the reviews by the number of reviews per product
def filter_by_number_reviews_product(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df_grouped = df[["parent_asin", "rating"]].groupby("parent_asin").count().reset_index()
    df_grouped.columns = ["parent_asin", "count"]
    df_filtered = df[df["parent_asin"].isin(df_grouped[df_grouped["count"] >= n]["parent_asin"])]
    return df_filtered


# intermediate________________________________________________________________________________________________
def preprocess_metadata(
    columns: list = []
) -> None:

    if not os.path.exists("data/_preprocessed/metadata.csv") or len(columns) > 0:
        os.makedirs("data/_preprocessed/") if not os.path.exists(
            "data/_preprocessed/"
        ) else None
        df = pd.read_json("data/_raw/meta_Software.jsonl", lines=True)
        
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Le seguenti colonne non esistono nel DataFrame: {missing_columns}")
        
        df = df.dropna(subset=['title'])
        df = df.dropna(subset=['description'])
        df['description'] = df['description'].astype(str)
        df = df[df['description'] != '[]']
        df = df[columns]
        df.to_csv("data/_preprocessed/metadata.csv", index=False)


# ___________________________________________________________________________________________________________
        
# This function filters the pre-processed reviews by date range and number of
# reviews per user/product
# and pre-processed metadata
def get_final():
    from collect_data import get_preprocessed_reviews, get_preprocessed_metadata
    
    print("""Preprocessing review dataset for Software and filtering
          by date number of reviews(user/product)...""")
    # Preprocess the reviews and save them to csv file
    preprocess_reviews(["rating", "parent_asin", "user_id", "timestamp"])
    # Load the preprocessed reviews
    df_ppr = get_preprocessed_reviews()
    # Filters
    df_ppr = filter_by_date_range(df_ppr, "2012-01-01", "2023-01-01")
    df_ppr = df_ppr[["user_id", "parent_asin", "rating"]]
    for _ in range(3):
        df_ppr = filter_by_number_reviews_user(df_ppr, 22)
        df_ppr = filter_by_number_reviews_product(df_ppr, 13)
    
    print("""Preprocessing metadata dataset for Software...""")
    preprocess_metadata(["parent_asin", "title", "description"])
    df_ppm = get_preprocessed_metadata()
    df_ppm = df_ppm[df_ppm["parent_asin"].isin(df_ppr["parent_asin"])]
    df_ppr = df_ppr[df_ppr["parent_asin"].isin(df_ppm["parent_asin"])]
    
    
    print(df_ppr["user_id"].nunique(), df_ppr["parent_asin"].nunique())

    os.makedirs("data/final/") if not os.path.exists("data/final/") else None
    df_ppr.to_csv("data/final/reviews.csv", index=False)
    df_ppm.to_csv("data/final/metadata.csv", index=False)

    
def main():
    get_final()


if __name__ == "__main__":
    main()
