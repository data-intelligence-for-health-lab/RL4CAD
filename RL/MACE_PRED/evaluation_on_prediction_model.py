import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sklearn as sk
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import training_constants as tc
from rl_utils import Episode, Transition, get_episodes, reward_func_mace, reward_func_mace_survival_repvasc
import d3rlpy
from torchinfo import summary
import torch
import datetime
from torch.optim import Adam
import argparse
from rl_utils import PolicyResolver, weighted_importance_sampling_with_bootstrap
from train_physician_imitation_model import PhysicianImitator
import torch
import torch.nn as nn
from train_mace_prediction_model import ModelBasedInference


reward = 'mace-survival-repvasc'
reward_func = reward_func_mace_survival_repvasc
maximize_outcome = True
reward_model_name = 'model_20240527235700.pth'
reward_model_path = os.path.join(tc.models_path, f'{reward}_prediction_model', reward_model_name)

# Load the data
processed_data_path = tc.processed_data_path
models_path = tc.models_path
rewards_list = tc.rewards_list

all_caths_train = pd.read_csv(os.path.join(processed_data_path, 'train', 'all_caths.csv'))
all_caths_train['SubsequentTreatment'].fillna('Medical Therapy', inplace=True)
treatment_train = all_caths_train['SubsequentTreatment']
all_caths_train_imputed = pd.read_csv(os.path.join(processed_data_path, 'train', 'all_caths_train_imputed.csv'))

all_caths_test = pd.read_csv(os.path.join(processed_data_path, 'test', 'all_caths.csv'))
all_caths_test['SubsequentTreatment'].fillna('Medical Therapy', inplace=True)
treatment_test = all_caths_test['SubsequentTreatment']
all_caths_test_imputed = pd.read_csv(os.path.join(processed_data_path, 'test', 'all_caths_test_imputed.csv'))


# encode the actions
treatments = ['Medical Therapy', 'PCI', 'CABG']
action_encoder = preprocessing.LabelEncoder()
action_encoder.fit(np.array(treatments).reshape(-1, 1))
actions_dict = dict(zip(action_encoder.classes_, action_encoder.transform(action_encoder.classes_)))
print("Actions dictionary:", actions_dict)

# Create the episodes
train_episodes = get_episodes(all_caths_train, all_caths_train_imputed, action_encoder, rewards_list, reward_func)
test_episodes = get_episodes(all_caths_test, all_caths_test_imputed, action_encoder, rewards_list, reward_func)


# behavior policy
best_n_clusters = tc.n_clusters_behavior_policy
best_physician_experiment_type = tc.experiment_type_behavior_policy
best_physician_policy = pickle.load(open(tc.behavior_policy_path, 'rb'))
best_physician_policy = best_physician_policy['physician']
train_clusters_path = os.path.join(tc.processed_data_path, 'train', f'{best_physician_experiment_type}', f'{best_physician_experiment_type}_kmeans_clusters_{best_n_clusters}.csv')
test_clusters_path = os.path.join(tc.processed_data_path, 'test', f'{best_physician_experiment_type}', f'{best_physician_experiment_type}_kmeans_clusters_{best_n_clusters}.csv')
all_caths_train_clusters = pd.read_csv(train_clusters_path)
all_caths_test_clusters = pd.read_csv(test_clusters_path)

behavior_train_episodes = get_episodes(all_caths_train, all_caths_train_clusters, action_encoder, rewards_list, reward_func)
behavior_test_episodes = get_episodes(all_caths_test, all_caths_test_clusters, action_encoder, rewards_list, reward_func)


# put the behavior policy inside the reference episodes
for dataset, behavior_dataset in [(train_episodes, behavior_train_episodes), (test_episodes, behavior_test_episodes)]:
    for episode, behavior_episode in zip(dataset, behavior_dataset):
        for transition, behavior_transition in zip(episode, behavior_episode):
            behavior_state = behavior_transition.state
            behavior_state = int(behavior_state[0])
            transition.prediction_probs['best_physician_policy'] = best_physician_policy[behavior_state, :]
        

# load the reward prediction model
reward_prediction_model = ModelBasedInference(reward_model_path, maximize_outcome=maximize_outcome)

# calculate the policy and store them in the episodes (to save time)
prediction_policy = PolicyResolver(reward_prediction_model, list(actions_dict.values()))
# put the mace prediction model inside the episodes
for dataset in [train_episodes, test_episodes]:
    for episode in dataset:
        for transition in episode:
            state = transition.state
            state = int(state[0])
            soft_probs = []
            for action in actions_dict.values():
                soft_probs.append(prediction_policy.predict_prob(transition, action))
            transition.prediction_probs['prediction_soft'] = np.array(soft_probs)
            transition.prediction_probs['prediction_greedy'] = np.zeros(len(actions_dict))
            transition.prediction_probs['prediction_greedy'][np.argmax(soft_probs)] = 1.0

# evaluate the model's policies
prediction_soft_policy = PolicyResolver('prediction_soft', list(actions_dict.values()))
prediction_greedy_policy = PolicyResolver('prediction_greedy', list(actions_dict.values()))
physician_policy = PolicyResolver('best_physician_policy', list(actions_dict.values()))

# evaluate the soft policy
wis, ci = weighted_importance_sampling_with_bootstrap(test_episodes, 0.99, prediction_soft_policy, physician_policy, num_bootstrap_samples=1000, N=1000)
print(f"Weighted importance sampling for the {reward} PREDICTION policy on the test data: {wis}, CI: {ci}")
wis, ci = weighted_importance_sampling_with_bootstrap(test_episodes, 0.99, physician_policy, physician_policy, num_bootstrap_samples=1000, N=1000)
print(f"Weighted importance sampling for the physician policy on the test data: {wis}, CI: {ci}")
wis, ci = weighted_importance_sampling_with_bootstrap(train_episodes, 0.99, prediction_soft_policy, physician_policy, num_bootstrap_samples=1000, N=1000)
print(f"Weighted importance sampling for the {reward} PREDICTION policy on the train data: {wis}, CI: {ci}")
wis, ci = weighted_importance_sampling_with_bootstrap(train_episodes, 0.99, physician_policy, physician_policy, num_bootstrap_samples=1000, N=1000)
print(f"Weighted importance sampling for the physician policy on the train data: {wis}, CI: {ci}")

# evaluate the greedy policy
wis, ci = weighted_importance_sampling_with_bootstrap(test_episodes, 0.99, prediction_greedy_policy, physician_policy, num_bootstrap_samples=1000, N=1000)
print(f"Weighted importance sampling for the GREEDY {reward} PREDICTION policy on the test data: {wis}, CI: {ci}")
wis, ci = weighted_importance_sampling_with_bootstrap(train_episodes, 0.99, prediction_greedy_policy, physician_policy, num_bootstrap_samples=1000, N=1000)
print(f"Weighted importance sampling for the GREEDY {reward} PREDICTION policy on the train data: {wis}, CI: {ci}")

# frequency of actions
reverse_actions_dict = {v: k for k, v in actions_dict.items()}
action_frequency = {}
for dataset in [test_episodes]:
    for episode in dataset:
        for transition in episode:
            action = np.argmax(transition.prediction_probs['prediction_greedy'])
            action_name = reverse_actions_dict[action]
            if action_name not in action_frequency:
                action_frequency[action_name] = 0
            action_frequency[action_name] += 1
            
print("Action frequency:", action_frequency)