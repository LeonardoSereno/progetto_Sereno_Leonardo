import os

# This function ensures that the directory for the given filepath exists.
def ensure_directory_exists(filepath: str) -> None:  
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

# This function reads the raw reviews data and prints the first n rows, columns, and shape of the DataFrame.
def get_reviews_raw_head(n: int = 5) -> None:
    import pandas as pd  
    df = pd.read_json("data/_raw/review_Software.jsonl", lines=True)
    print(df.head(n))
    print("\nColumns:", df.columns.tolist())
    print("\nShape:", df.shape)
    
def get_reviews_final_tail(n: int = 5) -> None:
    import pandas as pd  
    df = pd.read_csv("data/final/reviews.csv")
    print(df.tail(n))
    print("\nColumns:", df.columns.tolist())
    print("\nShape:", df.shape)
    
# Functions to describe data
def describe_reviews_per_product_raw() -> None:
    import pandas as pd

    df = pd.read_json("data/_raw/review_Software.jsonl", lines=True)

    df_grouped = df.groupby("parent_asin")["rating"].count().reset_index(name="count")
    
    stats = df_grouped["count"].describe()

    print("📦 Statistiche sul numero di recensioni per prodotto:")
    print(stats)

def find_product_by_review_count(target_count=50891):
    import pandas as pd

    df = pd.read_json("data/_raw/review_Software.jsonl", lines=True)

    review_counts = df.groupby('parent_asin').size().reset_index(name='count')

    matching_products = review_counts[review_counts['count'] == target_count]
    
    if not matching_products.empty:
        result = matching_products['parent_asin'].tolist()
        print(f"🎯 Trovati {len(result)} prodotti con {target_count} recensioni:")
        print(matching_products)
        return result
    else:
        closest = review_counts.iloc[(review_counts['count'] - target_count).abs().argsort()[:3]]
        print(f"❌ Nessun prodotto con ESATTAMENTE {target_count} recensioni")
        print("I 3 prodotti più vicini:")
        print(closest)
        return None

def describe_reviews_per_user_raw() -> None:
    import pandas as pd

    df = pd.read_json("data/_raw/review_Software.jsonl", lines=True)

    df_grouped = df.groupby("user_id")["rating"].count().reset_index(name="count")
    
    stats = df_grouped["count"].describe().apply(lambda x: f"{x:.2f}")

    print("📦 Statistiche sul numero di recensioni per user:")
    print(stats)

def describe_reviews_per_product_final() -> None:
    import pandas as pd

    df = pd.read_csv("data/final/reviews.csv")

    df_grouped = df.groupby("parent_asin")["rating"].count().reset_index(name="count")
    
    stats = df_grouped["count"].describe()

    print("📦 Statistiche sul numero di recensioni per prodotto:")
    print(stats)
    
    print("Head of sorted df grouped by user_id:")
    top_users = df_grouped.sort_values("count", ascending=False).head(10)
    print(top_users.to_string(index=False))

    
def describe_reviews_per_user_final() -> None:
    import pandas as pd

    df = pd.read_csv("data/final/reviews.csv")

    df_grouped = df.groupby("user_id")["rating"].count().reset_index(name="count")
    
    stats = df_grouped["count"].describe().apply(lambda x: f"{x:.2f}")

    print("📦 Statistiche sul numero di recensioni per user:")
    print(stats)
    
    print("Head of sorted df grouped by user_id:")
    top_users = df_grouped.sort_values("count", ascending=False).head(10)
    print(top_users.to_string(index=False))
    

# This function visualizes the distribution of NaN values across columns in the raw reviews dataset.
def get_plot_nans(saveFig=False, figName="imgs/nans_raw.png"):
    """Visualize NaN distribution across columns with improved formatting."""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_json("data/_raw/review_Software.jsonl", lines=True)

    nan_counts = df.isna().sum().sort_values(ascending=False)

    if nan_counts.sum() == 0:
        print("✅ No NaN values found in any column")
        return

    plt.figure(figsize=(12, 6))
    ax = nan_counts.plot(kind='bar', color='#1f77b4', edgecolor='black')

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}", 
            (p.get_x() + p.get_width()/2., p.get_height()),
            ha='center', va='center', 
            xytext=(0, 5), 
            textcoords='offset points'
        )

    plt.title("Missing Values Distribution", pad=20, fontsize=14)
    plt.xlabel("Columns", labelpad=10)
    plt.ylabel("NaN Count", labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    max_val = nan_counts.max()
    plt.ylim(0, max_val * 1.15)
    
    plt.tight_layout()
    
    if saveFig:
        ensure_directory_exists(figName)
        plt.savefig(figName, dpi=300, bbox_inches='tight')
    
    plt.show()


def get_ratings_distribution(saveFig=False, figName="imgs/ratings_dist_raw.png"):
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import os

    df = pd.read_json("data/_raw/review_Software.jsonl", lines=True)

    df_grouped = df.groupby("rating")["user_id"].count().reset_index()
    df_grouped.columns = ["rating", "count"]
    df_grouped = df_grouped.sort_values(by="rating")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(df_grouped["rating"], df_grouped["count"],
                   color="orange", edgecolor="black", linewidth=0.7)

    plt.title("Rating distribution", fontsize=16)
    plt.xlabel("Rating", fontsize=14)
    plt.ylabel("Number of users", fontsize=14)
    plt.ticklabel_format(style='plain')
    plt.xticks(df_grouped["rating"], fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()

    # Save plot
    if saveFig:
        os.makedirs(os.path.dirname(figName), exist_ok=True)
        plt.savefig(figName, dpi=300, bbox_inches="tight")

    plt.show()




def get_plot_time_distribution(saveFig=False, figName="imgs/plot_time_dist_raw.png"):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_json("data/_raw/review_Software.jsonl", lines=True)
    
    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    df_daily = df.groupby(df["timestamp"].dt.date).size().reset_index(name="n_reviews")
    df_daily["timestamp"] = pd.to_datetime(df_daily["timestamp"])
    df_daily = df_daily.set_index("timestamp")

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(df_daily.index, df_daily["n_reviews"], color="steelblue")
    plt.title("Reviews Distribution Over Time Grouped by Day")
    plt.xlabel("Year")
    plt.ylabel("Number of Reviews")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save plot
    if saveFig:
        os.makedirs(os.path.dirname(figName), exist_ok=True)
        plt.savefig(figName, dpi=300, bbox_inches="tight")

    plt.show()

'''
def get_boxplot_time_distribution(saveFig=False, figName="imgs/boxplot_time_dist_raw.png"):
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_json("data/_raw/review_Software.jsonl", lines=True)
    
    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year

    # Group by day and count reviews
    daily_counts = df.groupby(df["timestamp"].dt.date).size().reset_index(name="review_count")
    daily_counts["timestamp"] = pd.to_datetime(daily_counts["timestamp"])
    daily_counts["year"] = daily_counts["timestamp"].dt.year

    # Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=daily_counts, x="year", y="review_count")
    plt.title("Distribution of Reviews")
    plt.xlabel("Year")
    plt.ylabel("Dayly number of reviews")
    plt.grid(True)
    plt.tight_layout()

    if saveFig:
        os.makedirs(os.path.dirname(figName), exist_ok=True)
        plt.savefig(figName, dpi=300, bbox_inches='tight')

    plt.show()
    
'''  
    
    
def get_plot_nreviews_user(saveFig=False, figName="imgs/nreviews_user.png"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    df = pd.read_csv("data/_preprocessed/reviews.csv")

    df_grouped = df[["user_id", "rating"]].groupby("user_id").count().reset_index()
    df_grouped.columns = ["user_id", "count"]

    filters = list(range(2, 40))
    user_counts = [(df_grouped["count"] >= i).sum() for i in filters]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(filters, user_counts, marker='o', linestyle='-', color='steelblue', label="Utenti")
    plt.title("Number of users that exceed the review threshold")
    plt.xlabel("review threshold")
    plt.ylabel("Number of users")
    plt.xticks(filters)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if saveFig:
        os.makedirs(os.path.dirname(figName), exist_ok=True)
        plt.savefig(figName, dpi=300, bbox_inches="tight")

    plt.show()

def get_plot_nreviews_user_15(saveFig=False, figName="imgs/nreviews_user_15.png"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    df = pd.read_csv("data/_preprocessed/reviews.csv")

    df_grouped = df[["user_id", "rating"]].groupby("user_id").count().reset_index()
    df_grouped.columns = ["user_id", "count"]

    filters = list(range(15, 40))
    user_counts = [(df_grouped["count"] >= i).sum() for i in filters]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(filters, user_counts, marker='o', linestyle='-', color='steelblue', label="Utenti")
    plt.title("Number of users that exceed the review threshold")
    plt.xlabel("review threshold")
    plt.ylabel("Number of users")
    plt.xticks(filters)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if saveFig:
        os.makedirs(os.path.dirname(figName), exist_ok=True)
        plt.savefig(figName, dpi=300, bbox_inches="tight")

    plt.show()
    

def get_plot_nreviews_product(saveFig=False, figName="imgs/nreviews_product.png"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    df = pd.read_csv("data/_preprocessed/reviews.csv")

    df_grouped = df[["parent_asin", "rating"]].groupby("parent_asin").count().reset_index()
    df_grouped.columns = ["parent_asin", "count"]

    filters = list(range(2, 100))
    product_counts = [(df_grouped["count"] >= i).sum() for i in filters]

    plt.figure(figsize=(20, 8))
    plt.plot(filters, product_counts, marker='o', linestyle='-', color='darkorange', label="Prodotti")
    plt.title("Number of products that exceed the review threshold")
    plt.xlabel("review threshold")
    plt.ylabel("Number of products")
    plt.xticks(filters, rotation=90, fontsize=8)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if saveFig:
        os.makedirs(os.path.dirname(figName), exist_ok=True)
        plt.savefig(figName, dpi=300, bbox_inches='tight')

    plt.show()


    
def main():
    #get_reviews_raw_head(n=5)
    #get_reviews_final_tail(n=5)
    describe_reviews_per_product_raw()
    #find_product_by_review_count(50891)
    describe_reviews_per_user_raw()
    #describe_reviews_per_product_final()
    #describe_reviews_per_user_final()
    #get_plot_nans(saveFig=True, figName="imgs/nans_raw.png")
    #get_ratings_distribution(saveFig=True, figName="imgs/ratings_dist_raw.png")
    #get_plot_time_distribution(saveFig=True, figName="imgs/plot_time_dist_raw.png")
    '''get_boxplot_time_distribution(saveFig=True, figName="imgs/boxplot_time_dist_raw.png")'''#Function not in use
    #get_plot_nreviews_user(saveFig=True, figName="imgs/nreviews_user.png")
    #get_plot_nreviews_user_15(saveFig=True, figName="imgs/nreviews_user_15.png")
    #get_plot_nreviews_product(saveFig=True, figName="imgs/nreviews_product.png")
    
if __name__ == "__main__":
    main()