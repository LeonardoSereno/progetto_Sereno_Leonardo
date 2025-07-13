import pandas as pd, numpy as np, os
from sklearn.metrics import mean_squared_error

def transformers_create_embeddings_nomic(df):
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F

    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

    data = ["classification: " + text for text in df['title_desc'].tolist()]

    embeddings = model.encode(data, convert_to_tensor=True, batch_size=32, show_progress_bar=True)

    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = embeddings[:, :512]
    embeddings = F.normalize(embeddings, p=2, dim=1)

    embeddings_df = pd.DataFrame(embeddings.cpu().numpy())
    embeddings_df['parent_asin'] = df['parent_asin']

    # Salva CSV
    os.makedirs('data/transformers', exist_ok=True)
    embeddings_df.to_csv('data/transformers/transformers_embeddings.csv', index=False)

    return embeddings_df

def transformers_create_embeddings_minilm(df):
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    data = []
    
    for sent in df['title_desc'].to_list():
        data.append(sent)
        
    embeddings = model.encode(data)
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df['parent_asin'] = df['parent_asin']

    # Salva su disco
    os.makedirs('data/transformers', exist_ok=True)
    embeddings_df.to_csv('data/transformers/transformers_embeddings_minilmv12.csv', index=False)

    return embeddings_df
    
    
def transformers_knn(embeddings=None):
    import time
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor

    reviews_df = pd.read_csv('data/final/reviews.csv')

    # Carica dati di embedding
    if embeddings is None:
        transformers_embeddings_path = 'data/transformers/transformers_embeddings.csv'
        if not os.path.exists(transformers_embeddings_path):
            raise FileNotFoundError(f"Expected TF-IDF data at {transformers_embeddings_path}")
        transformers_embeddings = pd.read_csv(transformers_embeddings_path)
    else:
        transformers_embeddings = embeddings

    print("Start testing the transformers KNN model...")
    t = time.time()

    transformers_knn = []
    for user_id in reviews_df['user_id'].unique():
        user_reviews = reviews_df[reviews_df['user_id'] == user_id]
        rated_items = transformers_embeddings[transformers_embeddings['parent_asin'].isin(user_reviews['parent_asin'])]
        dataset = pd.merge(rated_items, user_reviews, on='parent_asin')
        dataset = dataset.drop(columns=['user_id'])
        try:
            X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns=['rating', 'parent_asin']), dataset['rating'], test_size=0.20)

            neigh_reg = KNeighborsRegressor(n_neighbors=12, metric="cosine")
            neigh_reg.fit(X_train, y_train)
            y_pred = neigh_reg.predict(X_test)
            transformers_knn.append(mean_squared_error(y_test, y_pred))
        except Exception as e:
            print("Error: ", e)

    mse = np.mean(transformers_knn)
    print("KNN results:")
    print(f"Mean Squared Error (KNN): {mse}")
    print(f"Root Mean Squared Error (KNN): {np.sqrt(mse)}")
    print(f"Time elapsed (KNN): {time.time()-t} seconds")
    
def execute_transformers_tecnique():
    meta_df = pd.read_csv('data/final/metadata.csv')
    if os.path.exists('data/lemmatized/lemmatized.csv'):
        lemmatized_df = pd.read_csv('data/lemmatized/lemmatized.csv')
    else:
        from nlp_preprocess import nlp_preprocess
        lemmatized_df = nlp_preprocess(meta_df.copy())

    embeddings_df_nomic = transformers_create_embeddings_nomic(lemmatized_df)
    embeddings_df_minilm = transformers_create_embeddings_minilm(lemmatized_df)
    transformers_knn(embeddings_df_nomic)
    transformers_knn(embeddings_df_minilm)


if __name__ == '__main__':
    execute_transformers_tecnique()
    