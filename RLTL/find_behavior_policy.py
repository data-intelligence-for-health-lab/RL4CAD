
import pandas as pd
import os
import numpy as np
import pickle
from datetime import datetime
from sklearn import preprocessing
import transfer_constants as trc
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import argparse

def calculate_clusters_accuracy(n_clusters, train_clusters, test_clusters, validation_clusters, treatment_train_encoded, treatment_test_encoded, treatment_validation_encoded):
    """
    This function processes the clusters and calculates the accuracy of the greedy physician

    Args:
    n_clusters: int, number of clusters
    train_clusters: pd.DataFrame, cluster assignments for the training data
    test_clusters: pd.DataFrame, cluster assignments for the testing data
    validation_clusters: pd.DataFrame, cluster assignments for the validation data
    treatment_train_encoded: np.array, encoded treatment assignments for the training data
    treatment_test_encoded: np.array, encoded treatment assignments for the testing data
    treatment_validation_encoded: np.array, encoded treatment assignments for the validation data

    """
    if not isinstance(train_clusters, pd.DataFrame):
        train_clusters = pd.DataFrame(train_clusters, columns=['cluster'])
    if not isinstance(test_clusters, pd.DataFrame):
        test_clusters = pd.DataFrame(test_clusters, columns=['cluster'])
    if not isinstance(validation_clusters, pd.DataFrame):
        validation_clusters = pd.DataFrame(validation_clusters, columns=['cluster'])

    
    train_clusters['real_treatment'] = treatment_train_encoded
    test_clusters['real_treatment'] = treatment_test_encoded
    validation_clusters['real_treatment'] = treatment_validation_encoded

    train_clusters['greedy_physician'] = np.zeros_like(treatment_train_encoded)
    test_clusters['greedy_physician'] = np.zeros_like(treatment_test_encoded)
    validation_clusters['greedy_physician'] = np.zeros_like(treatment_validation_encoded)

    for i in range(n_clusters):
        df_train_i = train_clusters[train_clusters['cluster'] == i]
        df_test_i = test_clusters[test_clusters['cluster'] == i]
        df_validation_i = validation_clusters[validation_clusters['cluster'] == i]

        # Get the most common treatment in the cluster
        if df_train_i.shape[0] != 0:
            most_common_treatment_train = df_train_i['real_treatment'].value_counts().idxmax()
            train_clusters.loc[train_clusters['cluster'] == i, 'greedy_physician'] = most_common_treatment_train
        if df_test_i.shape[0] != 0:
            most_common_treatment_test = df_test_i['real_treatment'].value_counts().idxmax()
            test_clusters.loc[test_clusters['cluster'] == i, 'greedy_physician'] = most_common_treatment_test
        if df_validation_i.shape[0] != 0:
            most_common_treatment_validation = df_validation_i['real_treatment'].value_counts().idxmax()
            validation_clusters.loc[validation_clusters['cluster'] == i, 'greedy_physician'] = most_common_treatment_validation

    # Calculate the accuracy of the greedy physician
    acc_train = accuracy_score(treatment_train_encoded, train_clusters['greedy_physician'])
    acc_test = accuracy_score(treatment_test_encoded, test_clusters['greedy_physician'])
    acc_validation = accuracy_score(treatment_validation_encoded, validation_clusters['greedy_physician'])

    return n_clusters, acc_train, acc_test, acc_validation

def train_kmeans_model_and_evaluate(n_clusters,
                                    all_caths_train_imputed,
                                    all_caths_test_imputed,
                                    all_caths_validation_imputed,
                                    treatment_train_encoded,
                                    treatment_test_encoded,
                                    treatment_validation_encoded,
                                    kmeans_model_path,
                                    kmeans_experiment_type):
    """
    This function trains a kmeans model and evaluates the accuracy of the greedy physician

    Args:
    n_clusters: int, number of clusters
    all_caths_train_imputed: pd.DataFrame, imputed training data
    all_caths_test_imputed: pd.DataFrame, imputed testing data
    all_caths_validation_imputed: pd.DataFrame, imputed validation data
    treatment_train_encoded: np.array, encoded treatment assignments for the training data
    treatment_test_encoded: np.array, encoded treatment assignments for the testing data
    treatment_validation_encoded: np.array, encoded treatment assignments for the validation data
    kmeans_model_path: str, path to save the kmeans model
    kmeans_experiment_type: str, type of kmeans experiment

    """
    max_iter = max(1000, 10 * n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=100, init='k-means++', n_init='auto', max_iter=max_iter).fit(all_caths_train_imputed)
    train_clusters = kmeans.predict(all_caths_train_imputed)
    test_clusters = kmeans.predict(all_caths_test_imputed)
    validation_clusters = kmeans.predict(all_caths_validation_imputed)

    # save the kmeans model
    kmeans_model_filename = os.path.join(kmeans_model_path, f'{kmeans_experiment_type}_{n_clusters}.pkl')
    with open(kmeans_model_filename, 'wb') as f:
        pickle.dump(kmeans, f)

    # calculate the accuracy of the greedy physician
    n_clusters, acc_train, acc_test, acc_validation = calculate_clusters_accuracy(n_clusters, train_clusters, test_clusters, validation_clusters, treatment_train_encoded, treatment_test_encoded, treatment_validation_encoded)

    return n_clusters, acc_train, acc_test, acc_validation
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the behavior policy for the stratified groups')
    parser.add_argument('--stratify_on', type=str, default='hospital', help='Stratify the patients based on the given feature')
    args = parser.parse_args()

    stratify_on = args.stratify_on
    groups = trc.stratification_consts[stratify_on]['groups']

    main_data_folder = trc.stratification_consts[stratify_on]['processed_data']
    main_models_folder = trc.stratification_consts[stratify_on]['models']
    # algo = 'CQL'
    # reward_function = reward_func_mace_survival_repvasc

    for group in groups:
        print(f"Start the job for {group} group at {datetime.now().strftime('%Y%m%d%H%M%S')}")
        data_folder = os.path.join(main_data_folder, group)
        models_folder = os.path.join(main_models_folder, group)
        
        # load the dataset
        all_caths_train = pd.read_csv(os.path.join(data_folder, 'train', 'all_caths.csv'))
        all_caths_train['SubsequentTreatment'].fillna('Medical Therapy', inplace=True)
        treatment_train = all_caths_train['SubsequentTreatment']
        all_caths_train_imputed = pd.read_csv(os.path.join(data_folder, 'train', 'all_caths_train_imputed.csv'))

        all_caths_test = pd.read_csv(os.path.join(data_folder, 'test', 'all_caths.csv'))
        all_caths_test['SubsequentTreatment'].fillna('Medical Therapy', inplace=True)
        treatment_test = all_caths_test['SubsequentTreatment']
        all_caths_test_imputed = pd.read_csv(os.path.join(data_folder, 'test', 'all_caths_test_imputed.csv'))

        all_caths_validation = pd.read_csv(os.path.join(data_folder, 'validation', 'all_caths.csv'))
        all_caths_validation['SubsequentTreatment'].fillna('Medical Therapy', inplace=True)
        treatment_validation = all_caths_validation['SubsequentTreatment']
        all_caths_validation_imputed = pd.read_csv(os.path.join(data_folder, 'validation', 'all_caths_validation_imputed.csv'))

        # drop feature containing the groups' information
        group_features_to_drop = trc.stratification_consts[stratify_on]['features_to_drop']
        all_caths_train.drop(columns=group_features_to_drop, inplace=True)
        all_caths_train_imputed.drop(columns=group_features_to_drop, inplace=True)
        all_caths_test.drop(columns=group_features_to_drop, inplace=True)
        all_caths_test_imputed.drop(columns=group_features_to_drop, inplace=True)
        all_caths_validation.drop(columns=group_features_to_drop, inplace=True)
        all_caths_validation_imputed.drop(columns=group_features_to_drop, inplace=True)

        # encode the actions
        treatments = trc.treatments
        action_encoder = preprocessing.LabelEncoder()
        action_encoder.fit(np.array(treatments).reshape(-1, 1))
        actions_dict = dict(zip(action_encoder.classes_, action_encoder.transform(action_encoder.classes_)))
        treatment_train_encoded = action_encoder.transform(treatment_train)
        treatment_test_encoded = action_encoder.transform(treatment_test)
        treatment_validation_encoded = action_encoder.transform(treatment_validation)
        print("Actions dictionary:", actions_dict)
        
        # kmeans clustering
        kmeans_experiment_type = 'vanilla_kmeans'
        kmeans_model_path = os.path.join(models_folder, f'{kmeans_experiment_type}_models')
        os.makedirs(kmeans_model_path, exist_ok=True)

        n_clusters_list = trc.n_clusters_list_cath
        print(f"Training {kmeans_experiment_type} models for the {group} group at {datetime.now().strftime('%Y%m%d%H%M%S')}")
        results = Parallel(n_jobs=-1)(delayed(train_kmeans_model_and_evaluate)(n_clusters, all_caths_train_imputed, all_caths_test_imputed, all_caths_validation_imputed, treatment_train_encoded, treatment_test_encoded, treatment_validation_encoded, kmeans_model_path, kmeans_experiment_type) for n_clusters in n_clusters_list)
        results_df = pd.DataFrame(results, columns=['n_clusters', 'acc_train', 'acc_test', 'acc_validation'])
        results_df.to_csv(os.path.join(models_folder, f'{kmeans_experiment_type}_accuracy_results.csv'), index=False)

        print(f"Finished training a {kmeans_experiment_type} model for the {group} group at {datetime.now().strftime('%Y%m%d%H%M%S')}")

        # find the best number of clusters (smallest n_clusters with the at least 98% of the maximum accuracy)
        max_acc_train = results_df['acc_train'].max()
        # behavior_n = results_df[(results_df['n_clusters']>100) & (results_df['acc_train'] >= 0.95 * max_acc_train)]['n_clusters'].min()
        behavior_n = results_df[results_df['acc_train'] >= 0.95 * max_acc_train]['n_clusters'].min()
        print(f"Best number of clusters for the {group} group: {behavior_n}")

        # calculate the behavior policy based on the best number of clusters
        behavior_policy_model_path = os.path.join(kmeans_model_path, f'{kmeans_experiment_type}_{behavior_n}.pkl')
        with open(behavior_policy_model_path, 'rb') as f:
            behavior_policy_model = pickle.load(f)
        train_clusters = behavior_policy_model.predict(all_caths_train_imputed)
        n_states = behavior_n
        n_actions = len(treatments)
        transition_matrix = np.zeros((n_states, n_actions))
        for i in range(n_states):
            idx_i = train_clusters == i
            for j in range(n_actions):
                idx_j = treatment_train_encoded == j
                transition_matrix[i, j] = np.sum(idx_i & idx_j)

        # calculate physician's policy (probability of each action for each state)
        sum_over_actions = np.sum(transition_matrix, axis=1)
        behavior_policy = np.divide(transition_matrix, sum_over_actions[:, None], out=np.zeros_like(transition_matrix), where=sum_over_actions[:, None] != 0)
        behavior_policy[sum_over_actions == 0, :] = 1 / n_actions
        greedy_physician_policy = np.zeros_like(behavior_policy)
        for i in range(n_states):
            greedy_physician_policy[i, np.argmax(transition_matrix[i, :])] = 1

        # save the behavior policy
        behavior_policy_filename = os.path.join(models_folder, f'behavior_policy_{kmeans_experiment_type}.pkl')

        behavior_policy_dict = {
            'n_clusters': behavior_n,
            'behavior_policy': behavior_policy,
            'greedy_physician_policy': greedy_physician_policy,
            'actions_dict': actions_dict
        }
        with open(behavior_policy_filename, 'wb') as f:
            pickle.dump(behavior_policy_dict, f)

        


    print("Finished training all models")