import pandas as pd

# Base project ________________________________________________________________

def get_preprocessed_reviews() -> pd.DataFrame:   
    import os
    if not os.path.exists("data/_preprocessed/reviews.csv"):
        raise FileNotFoundError(
            "The preprocessed dataset does not exist."
        )
    return pd.read_csv("data/_preprocessed/reviews.csv")

def get_preprocessed_metadata() -> pd.DataFrame:
    import os
    if not os.path.exists("data/_preprocessed/metadata.csv"):
        raise FileNotFoundError(
            "The preprocessed dataset does not exist."
        )
    return pd.read_csv("data/_preprocessed/metadata.csv")

def main():
    get_preprocessed_reviews()


if __name__ == "__main__":
    main()