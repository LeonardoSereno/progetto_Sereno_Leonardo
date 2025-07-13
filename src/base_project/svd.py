from surprise import SVD, model_selection
import json, os
import numpy as np
import pandas as pd

import json
import os
from surprise import SVD
from surprise import model_selection

def find_svd_config(dataset) -> dict:
    config_file = 'data/basic/svd.json'
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            data = json.load(f)
        print('Best SVD configuration loaded from file')
        print(f'Best SVD MSE = {data["mse"]:.4f}')
        print(f'Best SVD RMSE = {data["rmse"]:.4f}')
        print(f'Best SVD configuration = {data["config"]}')
        return data['mse'], data['rmse'], data['config']
    
    print('Searching best SVD configuration...')
    param_grid = {
        'n_factors': list(range(20, 200, 20)),
        'n_epochs': list(range(10, 70, 10)),
        'biased': [True, False]
    }

    gs = model_selection.GridSearchCV(SVD, param_grid,
                                    measures=["rmse", "mse"],
                                    cv=10,
                                    n_jobs=-1)
    gs.fit(dataset)

    print(f'Best SVD MSE = {gs.best_score["mse"]:.4f}')
    print(f'Best SVD RMSE = {gs.best_score["rmse"]:.4f}')
    print(f'Best SVD configuration = {gs.best_params["rmse"]}')
    
    # Salva i risultati
    os.makedirs('data/basic', exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump({
            'mse': gs.best_score["mse"],
            'rmse': gs.best_score["rmse"],
            'config': gs.best_params["rmse"]
        }, f)
    
    return gs.best_score["mse"], gs.best_score["rmse"], gs.best_params["rmse"]


def get_or_create_filled_svd_rating_matrix(dataset, users_id, items_id, config) -> pd.DataFrame:
    file_path = 'data/basic/filled_svd_rating_matrix.csv'
    
    if os.path.exists(file_path):
        filled_svd_rating_matrix = pd.read_csv(file_path, index_col=0)
        return filled_svd_rating_matrix
    
    print("Creating svd rating matrix")
    
    trainset = dataset.build_full_trainset()
    
    algo = SVD(n_factors=config['n_factors'],
               n_epochs=config['n_epochs'],
               biased=config['biased'])
    algo.fit(trainset)

    filled_svd_rating_matrix = []
    for uid in users_id:
        filled_svd_rating_matrix.append([])
        for iid in items_id:
            res = algo.predict(uid=uid, iid=iid)
            if res.r_ui is not None:
                filled_svd_rating_matrix[-1].append(0)
            else:
                filled_svd_rating_matrix[-1].append(res.est)

    filled_svd_rating_matrix = np.array(filled_svd_rating_matrix)

    os.makedirs('data/basic', exist_ok=True)
    
    filled_svd_rating_matrix = pd.DataFrame(filled_svd_rating_matrix, index=users_id, columns=items_id)
    filled_svd_rating_matrix.to_csv(file_path)
    
    print("rating matrix svd filled")

    return filled_svd_rating_matrix