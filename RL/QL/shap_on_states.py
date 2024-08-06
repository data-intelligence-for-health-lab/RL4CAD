import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import json
import training_constants as tc
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

def convert(o):
    if isinstance(o, np.float32):
        return float(o)
    raise TypeError

optimal_n_clusters = 84   # this is the best number of clusters found from previous experiments

action_dict = {0: 'CABG', 1: 'Medical Therapy', 2: 'PCI'}
experiment_name = 'vanilla_kmeans-2-999-all-caths-reward-mace-survival-repvasc__20240528-022243'
experiment_type = 'vanilla_kmeans'
load_existing_xgboost = True

policies_file = os.path.join(tc.models_path, f'{experiment_name}_rl_policies', f'policies_{optimal_n_clusters}.pkl')
policies = pickle.load(open(policies_file, 'rb'))

optimal_policy = policies['optimal']
physician_policy = policies['physician']

# load the kmeans model associated with the best clustering
kmeans_model_path = os.path.join(tc.models_path, f"{experiment_type}_models", f"kmeans_{experiment_type}_{optimal_n_clusters}.pkl")
kmeans_model = pickle.load(open(kmeans_model_path, 'rb'))


# load preprocessor to scale the centers back to original scale
agg_strategy_df = pd.read_csv('agg_strategy_v2.csv')
preprocessor_path = os.path.join(tc.models_path, f"preprocessor.pkl")
preprocessor = pickle.load(open(preprocessor_path, 'rb'))

# load the column names of the imputed data
imputed_data_path = os.path.join(tc.processed_data_path, 'train', 'all_caths_train_imputed.csv')
column_names = pd.read_csv(imputed_data_path, nrows=1).columns

# separate the numerical and categorical columns
non_categorical_cols = agg_strategy_df[agg_strategy_df['variable_type'].isin(['int', 'float', 'lab_float'])]['Variable_name'].values
non_categorical_cols = [x for x in non_categorical_cols if x in column_names]

# put the cluster centers in a dataframe
cluster_centers = pd.DataFrame(kmeans_model.cluster_centers_, columns=column_names)

# scale the cluster centers back to original scale
cluster_centers[non_categorical_cols] = preprocessor['scaler'].inverse_transform(cluster_centers[non_categorical_cols])

# binarize the categorical columns
categorical_cols = [x for x in column_names if x not in non_categorical_cols]
for col in categorical_cols:
    cluster_centers[col] = cluster_centers[col].round()

# add the optimal and physician actions to the cluster centers
cluster_centers['optimal_action'] = [action_dict[np.argmax(policy)] for policy in optimal_policy[:-1]]
cluster_centers['physician_action'] = [action_dict[np.argmax(policy)] for policy in physician_policy[:-1]]
cluster_centers = cluster_centers[['optimal_action', 'physician_action'] + column_names.tolist()]


# load the processed data
all_caths_train_imputed = pd.read_csv(os.path.join(tc.processed_data_path, 'train', 'all_caths_train_imputed.csv'))
all_caths_test_imputed = pd.read_csv(os.path.join(tc.processed_data_path, 'test', 'all_caths_test_imputed.csv'))

# get the cluster assignments for the train and test data
train_clusters_path = os.path.join(tc.processed_data_path, 'train', f'{experiment_type}', f'{experiment_type}_kmeans_clusters_{optimal_n_clusters}.csv')
train_clusters = pd.read_csv(train_clusters_path)['cluster'].values
test_clusters_path = os.path.join(tc.processed_data_path, 'test', f'{experiment_type}', f'{experiment_type}_kmeans_clusters_{optimal_n_clusters}.csv')
test_clusters = pd.read_csv(test_clusters_path)['cluster'].values

# fit xgboost model to predict the cluster assignment
X_train = all_caths_train_imputed
y_train = train_clusters
X_test = all_caths_test_imputed
y_test = test_clusters

# xgboost model paths
experiment_folder = os.path.join(tc.secondary_models_path, experiment_name, 'xgboost_on_states')
os.makedirs(experiment_folder, exist_ok=True)
best_model_path = os.path.join(experiment_folder, f'xgboost_model_on_states_{optimal_n_clusters}.pkl')

if load_existing_xgboost:
    best_model = pickle.load(open(best_model_path, 'rb'))
else:

        # Parameter grid
    param_grid = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 300],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

    # Configure the XGBClassifier and GridSearchCV
    model = xgb.XGBClassifier(tree_method='hist', objective='multi:softmax', n_jobs=40, device='cuda:1')
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=True, n_jobs=50)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best model parameters: {grid_search.best_params_}")

    # Save best model
    pickle.dump(best_model, open(best_model_path, 'wb'))


# Predictions
predictions = best_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)
print("Best model accuracy: ", accuracy)
conf_matrix = confusion_matrix(y_test, predictions)
conf_matrix = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 10))
plt.imshow(conf_matrix, cmap='viridis')
plt.colorbar()
plt.xticks(np.arange(optimal_n_clusters), np.arange(optimal_n_clusters), rotation=90)
plt.yticks(np.arange(optimal_n_clusters), np.arange(optimal_n_clusters))
plt.xlabel('Predicted cluster')
plt.ylabel('True cluster')
plt.title(f'Confusion matrix of the best model on states\nAccuracy: {accuracy:.2f}')
plt.savefig(os.path.join(experiment_folder, 'confusion_matrix.png'))

# SHAP values
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
print(f"SHAP values shape: {shap_values.shape}")

# save the name of the first 10 important features for each cluster + the abs shape value + the cluster center value
how_many_features = len(column_names)
top_features = {}
for cluster_num in range(optimal_n_clusters):
    cluster_center_row = cluster_centers.iloc[cluster_num]
    cluster_shap_values = shap_values[:, :, cluster_num]
    cluster_shap_values = np.abs(cluster_shap_values).mean(axis=0)
    top_features[cluster_num] = {}
    for feature_num in range(how_many_features):
        feature_idx = np.argmax(cluster_shap_values)
        feature_name = X_train.columns[feature_idx]
        feature_value = cluster_center_row[feature_name]
        shap_value = cluster_shap_values[feature_idx]
        top_features[cluster_num][feature_name] = {'feature_center': feature_value, 'shap_value': shap_value}
        cluster_shap_values[feature_idx] = 0

# save the top features
top_features_path = os.path.join(experiment_folder, f'top_{how_many_features}_features.json')
with open(top_features_path, 'w') as f:
    json.dump(top_features, f, indent=4, default=convert)

# save the SHAP values
shap_values_path = os.path.join(experiment_folder, f'shap_values_{optimal_n_clusters}.pkl')
pickle.dump(shap_values, open(shap_values_path, 'wb'))



