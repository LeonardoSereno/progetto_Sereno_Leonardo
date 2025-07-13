import os, numpy as np, json, pandas as pd
from surprise import KNNBasic, model_selection

def find_knn_config(dataset_surprise):
    file_path = 'data/basic/knn.jsonl'
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            result = json.loads(f.readline())
            return result["mse"], result["rmse"], result["params"]

    param_grid = {
        'k': list(range(10, 51, 2)),
        'sim_options': {
            'name': ['cosine', 'msd', 'pearson'],
            'user_based': [True, False],
        },
    }
    
    gs = model_selection.GridSearchCV(KNNBasic, param_grid,
                                      measures=["rmse", "mse"],
                                      cv=10,
                                      n_jobs=-1)
    
    gs.fit(dataset_surprise)

    best_mse = gs.best_score["mse"]
    best_rmse = gs.best_score["rmse"]
    best_params = gs.best_params["rmse"]

    print(f'Best MSE for KNN = {best_mse:.4f}')
    print(f'Best RMSE for KNN = {best_rmse:.4f}')
    print(f'Best configuration for KNN = {best_params}')

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump({
            "mse": best_mse,
            "rmse": best_rmse,
            "params": best_params
        }, f)
        f.write('\n')
    
    return best_mse, best_rmse, best_params



   
def get_or_create_filled_rating_matrix(dataset, users_id, items_id, best_config) -> pd.DataFrame:
    file_path = 'data/basic/filled_knn_rating_matrix.csv'
    
    # Check if file exists
    if os.path.exists(file_path):
        filled_rating_matrix = pd.read_csv(file_path, index_col=0)
        return filled_rating_matrix
    
    print('Creating new KNN filled rating matrix...')
    trainset = dataset.build_full_trainset()
    
    algo = KNNBasic(k=best_config['k'], sim_options=best_config['sim_options'])
    algo.fit(trainset)

    filled_rating_matrix = []
    for uid in users_id:
        filled_rating_matrix.append([])
        for iid in items_id:
            res = algo.predict(uid=uid, iid=iid)
            if res.r_ui is not None:
                filled_rating_matrix[-1].append(0)
            else:
                filled_rating_matrix[-1].append(res.est)

    filled_rating_matrix = np.array(filled_rating_matrix)
    
    os.makedirs('data/basic', exist_ok=True)

    filled_rating_matrix = pd.DataFrame(filled_rating_matrix, index=users_id, columns=items_id)
    filled_rating_matrix.to_csv(file_path)
    
    print("Filled rating matrix saved")

    return filled_rating_matrix
