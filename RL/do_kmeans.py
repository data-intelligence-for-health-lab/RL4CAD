import os
import pandas as pd
import numpy as np
import training_constants as tc
import pickle
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

def run_kmeans_and_save(n_clusters_list, data_df, dataset_name, processed_data_path, kmeans_folder, random_state=100, load_existing=False, verbose=False):
    '''
    run kmeans clustering on the data and save the clusters
    :param n_clusters_list: the list of number of clusters
    :param data_df: the dataframe that contains the data
    :param dataset_name: the name of the dataset (cath or checkup)
    :param processed_data_path: the path to save the clusters index
    :param kmeans_folder: the folder to save the kmeans models
    :return: None
    '''
    for n_clusters in n_clusters_list:
        kmeans_filename = f'kmeans_{dataset_name}_{n_clusters}.pkl'
        kmeans_filepath = os.path.join(kmeans_folder, kmeans_filename)
        if load_existing:
            with open(kmeans_filepath, 'rb') as f:
                kmeans = pickle.load(f)
        else:
            max_iter = max(500, 10 * n_clusters)
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, init='k-means++', n_init='auto', max_iter=max_iter).fit(data_df)
            with open(kmeans_filepath, 'wb') as f:
                pickle.dump(kmeans, f)

        clusters = kmeans.predict(data_df)
        clusters_df = pd.DataFrame(clusters, columns=['cluster'])
        clusters_df.to_csv(os.path.join(processed_data_path, f'{dataset_name}_kmeans_clusters_{n_clusters}.csv'), index=False)

        if verbose:
            print(f'K-means clustering for {dataset_name} with {n_clusters} clusters complete and file saved.')

def job_for_kmeans(n_clusters,
                      all_caths_train_imputed, all_caths_test_imputed, all_caths_validation_imputed,
                      read_clusters_from_file=False, experiment_type='vanilla_kmeans'):
    
    train_clusters_path = os.path.join(tc.processed_data_path, 'train', f'{experiment_type}', f'{experiment_type}_kmeans_clusters_{n_clusters}.csv')
    if not (os.path.exists(train_clusters_path) and read_clusters_from_file):
        run_kmeans_and_save([n_clusters], all_caths_train_imputed, f'{experiment_type}',
                            os.path.join(tc.processed_data_path, 'train', f'{experiment_type}'),
                            os.path.join(tc.models_path, f'{experiment_type}_models'),
                            load_existing=False, verbose=True)

    test_clusters_path = os.path.join(tc.processed_data_path, 'test', f'{experiment_type}', f'{experiment_type}_kmeans_clusters_{n_clusters}.csv')
    if not (os.path.exists(test_clusters_path) and read_clusters_from_file):
        run_kmeans_and_save([n_clusters], all_caths_test_imputed, f'{experiment_type}',
                            os.path.join(tc.processed_data_path, 'test', f'{experiment_type}'),
                            os.path.join(tc.models_path, f'{experiment_type}_models'),
                            load_existing=True, verbose=True)
        

    validation_clusters_path = os.path.join(tc.processed_data_path, 'validation', f'{experiment_type}', f'{experiment_type}_kmeans_clusters_{n_clusters}.csv')
    if not (os.path.exists(validation_clusters_path) and read_clusters_from_file):
        run_kmeans_and_save([n_clusters], all_caths_validation_imputed, f'{experiment_type}',
                            os.path.join(tc.processed_data_path, 'validation', f'{experiment_type}'),
                            os.path.join(tc.models_path, f'{experiment_type}_models'),
                            load_existing=True, verbose=True)

def do_kmeans():

    experiment_type = 'vanilla_kmeans'

    # folder to save the kmeans models and clusters
    kmeans_folder = os.path.join(tc.models_path, f'{experiment_type}_models')
    os.makedirs(kmeans_folder, exist_ok=True)
    os.makedirs(os.path.join(tc.processed_data_path, 'train', f'{experiment_type}'), exist_ok=True)  # to save the clustering results
    os.makedirs(os.path.join(tc.processed_data_path, 'test', f'{experiment_type}'), exist_ok=True)  # to save the clustering results
    os.makedirs(os.path.join(tc.processed_data_path, 'validation', f'{experiment_type}'), exist_ok=True)  # to save the clustering results


    # load the imputed data
    all_caths_train_imputed = pd.read_csv(os.path.join(tc.processed_data_path, 'train', 'all_caths_train_imputed.csv'))
    all_caths_test_imputed = pd.read_csv(os.path.join(tc.processed_data_path, 'test', 'all_caths_test_imputed.csv'))
    all_caths_validation_imputed = pd.read_csv(os.path.join(tc.processed_data_path, 'validation', 'all_caths_validation_imputed.csv'))

    # number of clusters to try
    n_clusters_list_cath = tc.n_clusters_list_cath

    # parallel run
    Parallel(n_jobs=-1)(delayed(job_for_kmeans)(n_clusters,
                                                all_caths_train_imputed, all_caths_test_imputed, all_caths_validation_imputed,
                                                read_clusters_from_file=False, experiment_type=experiment_type)
                        for n_clusters in n_clusters_list_cath)