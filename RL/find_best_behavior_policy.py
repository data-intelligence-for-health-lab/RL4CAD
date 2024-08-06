from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle
import os
import training_constants as tc
from sklearn import preprocessing
from matplotlib import pyplot as plt
import tqdm

experiment_type = 'vanilla_kmeans'
n_clusters_list = tc.n_clusters_list_cath
# processed_data_path = 'processed_data_obstructive_cad5'
# rl_policies_path = 'models_obstructive_cad5/vanilla_kmeans-2-999-all-caths-reward-mace-survival__20240426-002907_rl_policies'
processed_data_path = tc.processed_data_path
rl_policies_path = tc.models_path

# # Function to process each number of clusters
# def process_clusters(n_clusters, experiment_type, tc):
#     train_clusters_path = os.path.join(processed_data_path, 'train', f'{experiment_type}', f'{experiment_type}_kmeans_clusters_{n_clusters}.csv')
#     test_clusters_path = os.path.join(processed_data_path, 'test', f'{experiment_type}', f'{experiment_type}_kmeans_clusters_{n_clusters}.csv')

#     all_caths_train_clusters = pd.read_csv(train_clusters_path)
#     all_caths_test_clusters = pd.read_csv(test_clusters_path)

#     policies = pickle.load(open(os.path.join(rl_policies_path, f'policies_{n_clusters}.pkl'), 'rb'))

#     test_pred = policies['greedy_physician'][all_caths_test_clusters['cluster'].to_list()]
#     test_pred_encoded = np.argmax(test_pred, axis=1)
#     train_pred = policies['greedy_physician'][all_caths_train_clusters['cluster'].to_list()]
#     train_pred_encoded = np.argmax(train_pred, axis=1)

#     acc_test = accuracy_score(treatment_test_encoded, test_pred_encoded)
#     acc_train = accuracy_score(treatment_train_encoded, train_pred_encoded)

#     return n_clusters, acc_train, acc_test

# Function to process each number of clusters
def process_clusters(n_clusters, experiment_type, treatment_train_encoded, treatment_test_encoded):
    train_clusters_path = os.path.join(processed_data_path, 'train', f'{experiment_type}', f'{experiment_type}_kmeans_clusters_{n_clusters}.csv')
    test_clusters_path = os.path.join(processed_data_path, 'test', f'{experiment_type}', f'{experiment_type}_kmeans_clusters_{n_clusters}.csv')

    all_caths_train_clusters = pd.read_csv(train_clusters_path)
    all_caths_test_clusters = pd.read_csv(test_clusters_path)

    all_caths_train_clusters['real_treatment'] = treatment_train_encoded
    all_caths_test_clusters['real_treatment'] = treatment_test_encoded

    all_caths_train_clusters['greedy_physician'] = np.zeros_like(treatment_train_encoded)
    all_caths_test_clusters['greedy_physician'] = np.zeros_like(treatment_test_encoded)

    for i in range(n_clusters):
        df_train_i = all_caths_train_clusters[all_caths_train_clusters['cluster'] == i]
        df_test_i = all_caths_test_clusters[all_caths_test_clusters['cluster'] == i]

        # Get the most common treatment in the cluster
        if df_train_i.shape[0] != 0:
            most_common_treatment_train = df_train_i['real_treatment'].value_counts().idxmax()
            all_caths_train_clusters.loc[all_caths_train_clusters['cluster'] == i, 'greedy_physician'] = most_common_treatment_train
        if df_test_i.shape[0] != 0:
            most_common_treatment_test = df_test_i['real_treatment'].value_counts().idxmax()
            all_caths_test_clusters.loc[all_caths_test_clusters['cluster'] == i, 'greedy_physician'] = most_common_treatment_test

    # Calculate the accuracy of the greedy physician
    acc_train = accuracy_score(treatment_train_encoded, all_caths_train_clusters['greedy_physician'])
    acc_test = accuracy_score(treatment_test_encoded, all_caths_test_clusters['greedy_physician'])

    return n_clusters, acc_train, acc_test


# Encode the actions
action_encoder = preprocessing.LabelEncoder()
action_encoder.fit(np.array(tc.treatments))


all_caths_train = pd.read_csv(os.path.join(processed_data_path, 'train', 'all_caths.csv'))
all_caths_train['SubsequentTreatment'].fillna('Medical Therapy', inplace=True)
treatment_train = all_caths_train['SubsequentTreatment'].to_numpy()
all_caths_test = pd.read_csv(os.path.join(processed_data_path, 'test', 'all_caths.csv'))
all_caths_test['SubsequentTreatment'].fillna('Medical Therapy', inplace=True)
treatment_test = all_caths_test['SubsequentTreatment'].to_numpy()

# Apply the label encoder to the treatment column
treatment_train_encoded = action_encoder.transform(treatment_train)
treatment_test_encoded = action_encoder.transform(treatment_test)

# Parallel execution
results = Parallel(n_jobs=-1)(delayed(process_clusters)(n_clusters,
                                                        experiment_type,
                                                        treatment_train_encoded,
                                                        treatment_test_encoded) for n_clusters in tqdm.tqdm(n_clusters_list))

# Unpack results
n_clusters, all_acc_train, all_acc_test = zip(*results)

# Save the results
figure_path = os.path.join(tc.EXPERIMENTS_RESULTS, f'{experiment_type}_accuracy')
os.makedirs(figure_path, exist_ok=True)

# sort based on the number of clusters
df = pd.DataFrame({'n_clusters': n_clusters, 'acc_train': all_acc_train, 'acc_test': all_acc_test})
df = df.sort_values(by='n_clusters')
df.to_csv(os.path.join(figure_path, f"results_{experiment_type}.csv"), index=False)

# Plotting and results analysis
plt.figure(figsize=(10, 6))
plt.plot(df['n_clusters'], df['acc_train'], label='Train')
plt.plot(df['n_clusters'], df['acc_test'], label='Test')
plt.xlabel('Number of clusters')
plt.ylabel('Accuracy')
plt.legend()

max_acc_test = df['acc_test'].max()
max_n_clusters = 500
df2 = df[df['n_clusters'] <= max_n_clusters]

# find the smallest number of clusters with accuracy greater than 95% of the maximum accuracy
best_n_clusters = df2['n_clusters'][df2['acc_test'] > 0.95 * max_acc_test].min()

print(f'Best number of clusters: {best_n_clusters}')

plt.title(f"{experiment_type} - Best number of clusters: {best_n_clusters} with accuracy: {max(all_acc_test)}")

plt.savefig(os.path.join(figure_path, f"best_number_of_clusters_{experiment_type}.png"))