from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import os, json

def find_kmeans_config(dataset) -> dict:
    file_path = 'data/basic/kmeans.jsonl'
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            kmeans = json.load(f)
        print("Loading best configuration")
        print(f'KMEANS wcss = {kmeans["wcss"]:.4f}')
        print(f'KMEANS configuration = {kmeans["config"]}')
        
        return kmeans["wcss"], kmeans["config"]

    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        km.fit(dataset)
        wcss.append(km.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method', fontsize = 20)
    plt.xlabel('No. of Clusters')
    plt.ylabel('wcss')
    plt.show()

    n_clusters = int(input('Choose the best number of clusters: '))

    os.makedirs('data/basic', exist_ok=True)

    with open('data/basic/best_config_KMEANS.json', 'w') as f:
        json.dump({
            'wcss': wcss[n_clusters-1],
            'config': n_clusters
        }, f)

    return wcss[n_clusters-1], n_clusters