import pandas as pd
from surprise import Reader, Dataset
from knn import find_knn_config, get_or_create_filled_rating_matrix
from kmeans import find_kmeans_config
from svd import find_svd_config, get_or_create_filled_svd_rating_matrix

def sort_columns(row):
    sorted_columns = sorted(row.items(), key=lambda x: x[1], reverse=True)
    return [col[0] for col in sorted_columns[:5]]


def base_project(rating_matrix):
    reader = Reader(rating_scale=(1, 5))
    dataset_surprise = Dataset.load_from_df(
        rating_matrix[['user_id', 'parent_asin', 'rating']],
        reader
    )
    
    # 2 - Find the best KNN configuration
    knn_mse, knn_rmse, knn_rmse_params = find_knn_config(dataset_surprise)
    
    # 3 - Fill the rating matrix using KNN
    users_id = rating_matrix["user_id"].unique()
    items_id = rating_matrix["parent_asin"].unique()
    
    filled_rating_matrix = get_or_create_filled_rating_matrix(dataset_surprise, users_id, items_id, knn_rmse_params)
    
    # 4 - Clustering K-Means with cosine similarity
    kmeans_wcss, kmeans_config = find_kmeans_config(filled_rating_matrix)
    
    # 5 - Create raccomandation list for each user
    res_df = pd.DataFrame(filled_rating_matrix)
    res_df.columns = items_id
    res_df = res_df.set_index(users_id)

    rec_lists = pd.DataFrame(list(res_df.apply(sort_columns, axis=1)),index=res_df.index)
    
    print(rec_lists.head())
    
    # 6 -Compare the results knn - matrix factorization
    
    svd_mse, svd_rmse, svd_config = find_svd_config(dataset_surprise)
    
    get_or_create_filled_svd_rating_matrix(dataset_surprise, users_id, items_id, svd_config)
    
    print('\nComparison between KNN and SVD:')
    print('\tKNN\t|  SVD')
    print(f'MSE\t{knn_mse:.4f}  |  {svd_mse:.4f}')
    print(f'RMSE\t{knn_rmse:.4f}  |  {svd_rmse:.4f}')
    
    
if __name__ == "__main__":
    df = pd.read_csv("data/final/reviews.csv")
    base_project(df)