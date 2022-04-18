#from sklearn.metrics import mean_absolute_error, root_mean_square_error
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.knns import KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from statistics import mean
import pandas as pd
import os

print('Loading in the DataFrame')
df = pd.read_csv('ratings_small.csv')
print('Creating Reader')
reader = Reader(rating_scale=(1, 5))
print('Loading into Surprise Dataset')
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

'''
print('Creating models for task 3c and 3d')
pmf = SVD(biased=False, verbose=True)
knn_user = KNNBasic(verbose=True, sim_options={'user_based':'True'})
knn_item = KNNBasic(verbose=True, sim_options={'user_based':'False'})

print('PMF')
cross_validate(pmf, data, cv=5, measures=['MAE','RMSE'], verbose=True)

print('knn_user')
cross_validate(knn_user, data, cv=5, measures=['MAE','RMSE'], verbose=True)

print('knn_item')
cross_validate(knn_item, data, cv=5, measures=['MAE','RMSE'], verbose=True)

'''
'''
print('Creating models for task 3e')
knn_user = {method:KNNBasic(verbose=True, \
                            sim_options={'user_based':'True',\
                                         'name':method}) \
                    for method in ('msd', 'pearson', 'cosine')}
knn_item = {method:KNNBasic(verbose=True, \
                            sim_options={'user_based':'False',\
                                         'name':method}) \
                    for method in ('msd', 'pearson', 'cosine')}

for method, user_model in knn_user.items():
    print(f'User Model {method}')
    cross_validate(user_model, data, cv=5, measures=['MAE', 'RMSE'], verbose=True)

for method, item_model in knn_user.items():
    print(f'Item Model {method}')
    cross_validate(item_model, data, cv=5, measures=['MAE', 'RMSE'], verbose=True)
'''

print('Creating models for task 3f')

results = {'user':{'k':[],'rmse':[], 'mae':[]},\
           'item':{'k':[],'rmse':[], 'mae':[]}}
for k in range(0,100):
    print(f'User {k} | ', end='')
    user_based = KNNBasic(k=k, \
                          verbose=False, \
                          sim_options={'user_based':'True'})
    # Cross validate with user-based model
    dct = cross_validate(user_based, data, cv=5, measures=['MAE', 'RMSE'], n_jobs=-1)

    print(f'rmse: ' + str(float(mean(dct['test_rmse']))) + \
          f'\tmae: ' + str(float(mean(dct['test_mae']))))

    results['user']['k'].append(k)
    results['user']['rmse'].append(float(mean(dct['test_rmse'])))
    results['user']['mae'].append(float(mean(dct['test_mae'])))

    print(f'item {k} | ', end='')
    item_based = KNNBasic(k=k, \
                          verbose=False, \
                          sim_options={'user_based':'False'})
    # Cross validate with item based
    dct = cross_validate(item_based, data, cv=5, measures=['MAE', 'RMSE'], n_jobs=-1)

    print(f'rmse: ' + str(float(mean(dct['test_rmse']))) + \
          f'\tmae: ' + str(float(mean(dct['test_mae']))))

    # Add Results
    results['item']['k'].append(k)
    results['item']['rmse'].append(float(mean(dct['test_rmse'])))
    results['item']['mae'].append(float(mean(dct['test_mae'])))

with open('results_user.csv', 'w') as F:
    F.write('k,rmse,mae\n')
    k, rmse, mae = \
        results['user']['k'], results['user']['rmse'], results['user']['mae']
    F.writelines([f'{_k},{_rmse},{_mae}\n' for _k, _rmse, _mae in zip(k,rmse,mae)])

with open('results_item.csv', 'w') as F:
    F.write('k,rmse,mae\n')
    k, rmse, mae = \
        results['item']['k'], results['item']['rmse'], results['item']['mae']
    F.writelines([f'{_k},{_rmse},{_mae}\n' for _k, _rmse, _mae in zip(k,rmse,mae)])
