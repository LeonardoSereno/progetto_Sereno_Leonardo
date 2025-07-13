import pandas as pd, os, numpy as np
from sklearn.metrics import mean_squared_error

def tfidf_create_embeddings(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import os

    vectorizer = TfidfVectorizer()

    tfidf_model = vectorizer.fit_transform(df['title_desc'])

    tfidf_data = pd.DataFrame(tfidf_model.toarray(), columns=vectorizer.get_feature_names_out())

    tfidf_data['parent_asin'] = df['parent_asin']

    os.makedirs('data/tf_idf', exist_ok=True)
    tfidf_data.to_csv('data/tf_idf/tf_idf_data.csv', index=False)
    
    return tfidf_data
    
def tfidf_knn(embeddings=None):
    import time
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor

    reviews_df = pd.read_csv('data/final/reviews.csv')
    reviews_df = reviews_df.groupby('user_id').filter(lambda x: len(x) > 13)

    # Carica dati di embedding
    if embeddings is None:
        tfidf_path = 'data/tf_idf/tf_idf_data.csv'
        if not os.path.exists(tfidf_path):
            raise FileNotFoundError(f"Expected TF-IDF data at {tfidf_path}")
        tfidf_data = pd.read_csv(tfidf_path)
    else:
        tfidf_data = embeddings

    print("Start testing the tfidf KNN model...")
    t = time.time()

    tfidf_knn = []
    for user_id in reviews_df['user_id'].unique():
        user_reviews = reviews_df[reviews_df['user_id'] == user_id]
        rated_items = tfidf_data[tfidf_data['parent_asin'].isin(user_reviews['parent_asin'])]
        dataset = pd.merge(rated_items, user_reviews, on='parent_asin')
        dataset = dataset.dropna()
        dataset = dataset.drop(columns=['user_id'])

        try:
            X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns=['rating_y', 'parent_asin']), dataset['rating_y'], test_size=0.20)

            neigh_reg = KNeighborsRegressor(n_neighbors=12, metric="cosine")
            neigh_reg.fit(X_train, y_train)
            y_pred = neigh_reg.predict(X_test)
            tfidf_knn.append(mean_squared_error(y_test, y_pred))
        except Exception as e:
            print("Error tfidf KNN: ", e)
            continue

    print("tfidf(KNN) results:")
    print(f"MSE: {np.mean(tfidf_knn)}")
    print(f"RMSE: {np.sqrt(np.mean(tfidf_knn))}")
    print(f"Time tfidf(KNN): {time.time()-t} seconds")
    
    
def execute_tfidf_tecnique():
    meta_df = pd.read_csv('data/final/metadata.csv')
    if os.path.exists('data/lemmatized/lemmatized.csv'):
        lemmatized_df = pd.read_csv('data/lemmatized/lemmatized.csv')
    else:
        from nlp_preprocess import nlp_preprocess
        lemmatized_df = nlp_preprocess(meta_df.copy())
    
    tfidf_embeddings = tfidf_create_embeddings(lemmatized_df)
    tfidf_knn(tfidf_embeddings)

if __name__ == '__main__':
    execute_tfidf_tecnique()
