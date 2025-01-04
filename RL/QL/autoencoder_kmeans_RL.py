import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sklearn as sk
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import training_constants as tc
from rl_utils import Episode, Transition, get_episodes, ClusteringBasedInference
from rl_utils import reward_func_mace, reward_func_mace_survival_cost, reward_func_mace_survival, reward_func_mace_survival_repvasc_cost, reward_func_mace_survival_repvasc
import datetime
from torch.optim import Adam
import mdptoolbox
from rl_utils import PolicyResolver, weighted_importance_sampling_with_bootstrap
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import functools
from QL.autoencoder_sigmoid_training import Autoencoder
from do_kmeans import run_kmeans_and_save
import joblib
from joblib import Parallel, delayed
import argparse

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        

def do_q_learning_and_evaluate_policy(n_clusters,
                                      all_caths_train, all_caths_test,
                                      all_caths_train_clusters, all_caths_test_clusters,
                                      train_episodes_raw, test_episodes_raw, reward_function, only_train=False):

    # Load the data
    behavior_plicy_name = 'behavior_physician_policy'
    processed_data_path = tc.processed_data_path
    models_path = tc.models_path
    rewards_list = tc.rewards_list

    # load the model
    # clustering_model = pickle.load(open(os.path.join(models_path, 'kmeans_models', f'kmeans_cath_{n_clusters}.pkl'), 'rb'))

    # all_caths_train_clusters = pd.read_csv(os.path.join(processed_data_path, 'train', f'cath_kmeans_clusters_{n_clusters}.csv'))

    # all_caths_test = pd.read_csv(os.path.join(processed_data_path, 'test', 'all_caths.csv'))
    # all_caths_test['SubsequentTreatment'].fillna('Medical Therapy', inplace=True)
    # all_caths_test_clusters = pd.read_csv(os.path.join(processed_data_path, 'test', f'cath_kmeans_clusters_{n_clusters}.csv'))

    # encode the actions
    action_encoder = preprocessing.LabelEncoder()
    action_encoder.fit(np.array(tc.treatments))
    actions_dict = dict(zip(action_encoder.classes_, action_encoder.transform(action_encoder.classes_)))
    # print("Actions dictionary:", actions_dict)

    # Create the episodes
    train_episodes = get_episodes(all_caths_train, all_caths_train_clusters, action_encoder, rewards_list, reward_function)
    test_episodes = get_episodes(all_caths_test, all_caths_test_clusters, action_encoder, rewards_list, reward_function)


    # Create transition matrix
    n_actions = len(actions_dict)
    transition_matrix = np.zeros((n_actions, n_clusters + 1, n_clusters + 1), dtype=float)
    reward_matrix = np.zeros((n_actions, n_clusters + 1, n_clusters + 1), dtype=float)

    for episode in train_episodes:
        if len(episode) == 1:  # only one transition (reached to the terminal state)
            transition_matrix[episode.transitions[0].action, episode.transitions[0].state[0], n_clusters] += 1
            reward_matrix[episode.transitions[0].action, episode.transitions[0].state[0], n_clusters] += episode.transitions[0].reward
        else:
            for i in range(len(episode) - 1):
                transition_matrix[episode.transitions[i].action, episode.transitions[i].state[0], episode.transitions[i + 1].state[0]] += 1
                reward_matrix[episode.transitions[i].action, episode.transitions[i].state[0], episode.transitions[i + 1].state[0]] += episode.transitions[i].reward
            # last transition
            transition_matrix[episode.transitions[-1].action, episode.transitions[-1].state[0], n_clusters] += 1
            reward_matrix[episode.transitions[-1].action, episode.transitions[-1].state[0], n_clusters] += episode.transitions[-1].reward

    # calculate physician's policy (probability of each action for each state)
    sum_over_states = transition_matrix.sum(axis=2)
    sum_over_actions = sum_over_states.sum(axis=0)
    sum_over_states[:, sum_over_actions == 0] = 1  # set the states with no actions to 1 to avoid division by 0
    sum_over_actions = sum_over_states.sum(axis=0, keepdims=True)
    physician_policy = sum_over_states / sum_over_actions
    physician_policy = physician_policy.T

    # make sure that the transition matrix is stochastic (sum of each row is 1)
    # then normalize the rows of the transition matrix
    for i in range(transition_matrix.shape[0]):
        unreachable_states = np.where(transition_matrix[i,:,:].sum(axis=1) == 0)[0]
        transition_matrix[i, unreachable_states, :] = 1.0
        transition_matrix[i, :, :] = transition_matrix[i, :, :] / transition_matrix[i, :, :].sum(axis=1, keepdims=True)

    # Initialize Q-Learning
    ql = mdptoolbox.mdp.QLearning(transition_matrix, reward_matrix, discount=0.99, n_iter=1000000 * int(1 + n_clusters/100))

    # Run Q-Learning
    ql.run()

    # Get the optimal policy and Q-matrix
    optimal_policy = ql.policy
    Q_matrix = ql.Q


    # calculate the optimal policy probabilities
    Q_matrix_sum = np.sum(Q_matrix, axis=1, keepdims=True)
    Q_matrix_sum[Q_matrix_sum == 0] = 1  # set the states with no actions to 1 to avoid division by 0
    optimal_policy_probs = Q_matrix / Q_matrix_sum


    # calculate the greedy policy for the physician
    greedy_physician_policy = np.zeros((len(optimal_policy), len(actions_dict)))
    for i in range(physician_policy.shape[0]):
        greedy_physician_policy[i, np.argmax(physician_policy[i, :])] = 1.0

    # calculate the greedy policy for the optimal policy
    greedy_optimal_policy = np.zeros((len(optimal_policy), len(actions_dict)))
    for i in range(optimal_policy_probs.shape[0]):
        greedy_optimal_policy[i, np.argmax(optimal_policy_probs[i, :])] = 1.0

    # save the policies to a file
    policies = {'physician': physician_policy, 'optimal': optimal_policy_probs, 'greedy_physician': greedy_physician_policy}

    if only_train:
        return None, policies


    # EVALUATION with the Behavior Policy: Same Clustering Model
    # resolve the policies
    physician_policy_resolver = PolicyResolver(physician_policy, list(actions_dict.values()))
    greedy_physician_policy_resolver = PolicyResolver(greedy_physician_policy, list(actions_dict.values()))
    optimal_policy_resolver = PolicyResolver(optimal_policy_probs, list(actions_dict.values()))
    greedy_optimal_policy_resolver = PolicyResolver(greedy_optimal_policy, list(actions_dict.values()))

    # Evaluate the policies
    results = {}
    for dataset_name, dataset in [('test', test_episodes), ('train', train_episodes)]:
        for eval_policy_name, eval_policy_resolver in [('physician', physician_policy_resolver),
                                                       ('greedy_physician', greedy_physician_policy_resolver),
                                                       ('optimal', optimal_policy_resolver),
                                                       ('greedy_optimal', greedy_optimal_policy_resolver)]:
            wis, ci = weighted_importance_sampling_with_bootstrap(dataset, 0.99, eval_policy_resolver, physician_policy_resolver,
                                                                  num_bootstrap_samples=1000, N=1000, confidedence_level=0.95)
            results[f"{eval_policy_name}_{dataset_name}"] = [wis, ci[0], ci[1]]

    # print('---------------------------------------------------------------------------')
    # EVALUATION with the Behavior Policy: Best Clustering Model for the Physician Imitation

    current_physician_policy_probs = policies['physician']
    current_physician_policy_greedy_probs = policies['greedy_physician']

    # make greedy optimal policy
    greedy_optimal_policy_probs = np.zeros((optimal_policy_probs.shape[0], optimal_policy_probs.shape[1]))
    greedy_optimal_policy_probs[np.arange(optimal_policy_probs.shape[0]), np.argmax(optimal_policy_probs, axis=1)] = 1.0    
    
    # put the best physician policy inside the episodes
    for dataset, raw_dataset in [(train_episodes, train_episodes_raw), (test_episodes, test_episodes_raw)]:
        for ep, episode in enumerate(dataset):
            raw_episode = raw_dataset[ep]
            for tr, transition in enumerate(episode):
                state = transition.state
                state = int(state[0])
                transition.prediction_probs['optimal'] = optimal_policy_probs[state, :] # put the optimal policy inside the episode
                transition.prediction_probs['greedy_optimal'] = greedy_optimal_policy_probs[state, :] # put the greedy optimal policy inside the episode
                transition.prediction_probs['current_physician_policy'] = current_physician_policy_probs[state, :] # put the current physician policy inside the episode
                transition.prediction_probs['current_greedy_physician_policy'] = current_physician_policy_greedy_probs[state, :] # put the current physician policy inside the episode
                transition.prediction_probs[behavior_plicy_name] = raw_episode.transitions[tr].prediction_probs[behavior_plicy_name] # put the physician's policy inside the episode



    # # evaluate using the imitator model as the behavior policy
    behavior_plicy_resolver = PolicyResolver(behavior_plicy_name, list(actions_dict.values()))
    for dataset_name, dataset in [('test', test_episodes), ('train', train_episodes)]:
        for eval_policy_name in ['optimal', 'greedy_optimal', 'current_physician_policy', 'current_greedy_physician_policy']:
            eval_policy_resolver = PolicyResolver(eval_policy_name, list(actions_dict.values()))
            wis, ci = weighted_importance_sampling_with_bootstrap(dataset, 0.99, eval_policy_resolver, behavior_plicy_resolver,
                                                                  num_bootstrap_samples=1000, N=1000, confidedence_level=0.95)
            results[f"{eval_policy_name}_{dataset_name}_vs_{behavior_plicy_name}"] = [wis, ci[0], ci[1]]

    return results, policies


def eval_job_for_autoencoder_kmeans(n_clusters,
                      all_caths_train, all_caths_test,
                      train_episodes_raw, test_episodes_raw,
                      all_caths_train_imputed, all_caths_test_imputed, reward_function,
                      read_clusters_from_file=False, experiment_type='autoencoder_kmeans'):
    
    train_clusters_path = os.path.join(tc.processed_data_path, 'train', f'{experiment_type}', f'{experiment_type}_kmeans_clusters_{n_clusters}.csv')
    if not (os.path.exists(train_clusters_path) and read_clusters_from_file):
        run_kmeans_and_save([n_clusters], all_caths_train_imputed, f'{experiment_type}',
                            os.path.join(tc.processed_data_path, 'train', f'{experiment_type}'),
                            os.path.join(tc.models_path, f'{experiment_type}_models'),
                            load_existing=False, verbose=True)
        
    all_caths_train_clusters = pd.read_csv(train_clusters_path)

    test_clusters_path = os.path.join(tc.processed_data_path, 'test', f'{experiment_type}', f'{experiment_type}_kmeans_clusters_{n_clusters}.csv')
    if not (os.path.exists(test_clusters_path) and read_clusters_from_file):
        run_kmeans_and_save([n_clusters], all_caths_test_imputed, f'{experiment_type}',
                            os.path.join(tc.processed_data_path, 'test', f'{experiment_type}'),
                            os.path.join(tc.models_path, f'{experiment_type}_models'),
                            load_existing=True, verbose=True)
        
    all_caths_test_clusters = pd.read_csv(test_clusters_path)

    # validation_clusters_path = os.path.join(tc.processed_data_path, 'validation', f'{experiment_type}', f'{experiment_type}_kmeans_clusters_{n_clusters}.csv')
    # if not (os.path.exists(validation_clusters_path) and read_clusters_from_file):
    #     run_kmeans_and_save([n_clusters], all_caths_validation_imputed, f'{experiment_type}',
    #                         os.path.join(tc.processed_data_path, 'validation', f'{experiment_type}'),
    #                         os.path.join(tc.models_path, f'{experiment_type}_models'),
    #                         load_existing=True, verbose=True)

    # all_caths_validation_clusters = pd.read_csv(validation_clusters_path)


    results,policies = do_q_learning_and_evaluate_policy(n_clusters,
                                                all_caths_train, all_caths_test,
                                                all_caths_train_clusters, all_caths_test_clusters,
                                                train_episodes_raw, test_episodes_raw, reward_function)
    
    # policies_file = os.path.join(tc.models_path, f'{experiment_type}_models', f'policies_{n_clusters}.pkl')
    policies_file = os.path.join(tc.models_path, f'{experiment_name}_rl_policies', f'policies_{n_clusters}.pkl')
    pickle.dump(policies, open(policies_file, 'wb'))
    
    results_row = {'n_clusters': n_clusters}
    for key, value in results.items():
        results_row[key] = value[0]
        results_row[key + '_LB'] = value[1]
        results_row[key + '_UB'] = value[2]
    
    
    print(f"Finished evaluating the policies for number of clusters = {n_clusters}")
    return results_row


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the clustering-based inference using Q-Learning')
    parser.add_argument('--experiment_type', type=str, default='vanilla_kmeans', help='The type of the experiment (autoencoder_kmeans, autoencoder_sigmoid_8_kmeans, vanilla_kmeans)')
    parser.add_argument('--autoencoder_name', type=str, default=None, help='The name of the autoencoder model')
    parser.add_argument('--reward_name', type=str, default='mace-survival-repvasc', help='The name of the reward function')

    args = parser.parse_args()
    experiment_type = args.experiment_type
    autoencoder_name = args.autoencoder_name

    reward_name = args.reward_name
    if reward_name == 'mace':
        reward_function = reward_func_mace
    elif reward_name == 'mace-survival':
        reward_function = reward_func_mace_survival
    elif reward_name == 'mace-survival-repvasc':
        reward_function = reward_func_mace_survival_repvasc

    n_clusters_list = tc.n_clusters_list_cath
    # n_clusters_list = range(100, 120)
    # experiment_type = 'autoencoder_sigmoid_8_kmeans'
    # autoencoder_name = 'autoencoder_sigmoid_8'
    experiment_name = f'{experiment_type}-{n_clusters_list[0]}-{n_clusters_list[-1]}-all-caths' + '-reward-' + reward_name + '__' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # autoencoder loading
    if autoencoder_name is not None:
        autoencoder_model_path = os.path.join(tc.models_path, autoencoder_name, 'checkpoint.pth')
        checkpoint = torch.load(autoencoder_model_path)
        params = checkpoint['parameters']
        autoencoder = Autoencoder(params['input_dim'], params['latent_dim'], params['num_hidden_layers'], params['activation_fn'])
        autoencoder.load_state_dict(checkpoint['state_dict'])
        autoencoder = autoencoder.to(device)
        autoencoder.eval()
        print(f"Loaded the autoencoder model from {autoencoder_model_path}")

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
    treatments = tc.treatments
    action_encoder = preprocessing.LabelEncoder()
    action_encoder.fit(np.array(treatments).reshape(-1, 1))
    actions_dict = dict(zip(action_encoder.classes_, action_encoder.transform(action_encoder.classes_)))
    print("Actions dictionary:", actions_dict)

    # Create the episodes
    train_episodes_raw = get_episodes(all_caths_train, all_caths_train_imputed, action_encoder, rewards_list, reward_function)
    test_episodes_raw = get_episodes(all_caths_test, all_caths_test_imputed, action_encoder, rewards_list, reward_function)

    # best clustering for physician's policy to be used as the behavior policy
    behavior_experiment_type = tc.experiment_type_behavior_policy
    behavior_n_clusters = tc.n_clusters_behavior_policy

    # run the q-learning for the best clustering model for the physician imitation
    # TODO: It is not necessary to do Q-learning for the behavior policy.
    # TODO: Since I calculated the physician's policy within that function, I use that.
    # TODO: The better way is to have a different function for this. Maybe later.
    behavior_train_clusters_path = os.path.join(tc.processed_data_path, 'train', f'{behavior_experiment_type}', f'{behavior_experiment_type}_kmeans_clusters_{behavior_n_clusters}.csv')
    behavior_test_clusters_path = os.path.join(tc.processed_data_path, 'test', f'{behavior_experiment_type}', f'{behavior_experiment_type}_kmeans_clusters_{behavior_n_clusters}.csv')
    behavior_train_clusters = pd.read_csv(behavior_train_clusters_path)
    behavior_test_clusters = pd.read_csv(behavior_test_clusters_path)

    _,behavior_physician_policy = do_q_learning_and_evaluate_policy(behavior_n_clusters,
                                                all_caths_train, all_caths_test,
                                                behavior_train_clusters, behavior_test_clusters,
                                                None, None, reward_function, only_train=True)

    # behavior_physician_policy = pickle.load(open(os.path.join(tc.models_path, f'{behavior_experiment_type}_models', f'policies_{behavior_n_clusters}.pkl'), 'rb'))
    behavior_physician_policy = behavior_physician_policy['physician']

    # train_clusters_path = os.path.join(tc.processed_data_path, 'train', f'{experiment_type}', f'{experiment_type}_kmeans_clusters_{behavior_n_clusters}.csv')
    # test_clusters_path = os.path.join(tc.processed_data_path, 'test', f'{experiment_type}', f'{experiment_type}_kmeans_clusters_{behavior_n_clusters}.csv')
    # all_caths_train_clusters = pd.read_csv(train_clusters_path)
    # all_caths_test_clusters = pd.read_csv(test_clusters_path)

    behavior_train_episodes = get_episodes(all_caths_train, behavior_train_clusters, action_encoder, rewards_list, reward_function)
    behavior_test_episodes = get_episodes(all_caths_test, behavior_test_clusters, action_encoder, rewards_list, reward_function)
    # behavior_validation_episodes = get_episodes(all_caths_validation, behavior_validation_clusters, action_encoder, rewards_list, reward_function)

    # put the best physician policy, random policy, and single action policies inside the reference episodes
    # for dataset in [train_episodes_raw, test_episodes_raw]:
    for dataset, behavior_dataset in [(train_episodes_raw, behavior_train_episodes), (test_episodes_raw, behavior_test_episodes)]:
        for episode, behavior_episode in zip(dataset, behavior_dataset):
            for transition, behavior_transition in zip(episode, behavior_episode):
                behavior_state = behavior_transition.state
                behavior_state = int(behavior_state[0])
                transition.prediction_probs['behavior_physician_policy'] = behavior_physician_policy[behavior_state, :]
                
                # random action policies
                random_action = np.random.choice(list(actions_dict.values()))
                transition.prediction_probs['random'] = np.zeros(len(actions_dict))
                transition.prediction_probs['random'][random_action] = 1.0

                # single action policies
                for action_name, action_id in actions_dict.items():
                    transition.prediction_probs[action_name] = np.zeros(len(actions_dict))
                    transition.prediction_probs[action_name][action_id] = 1.0


    # if experiment_type == 'vanilla_kmeans':
    #     parallel_eval_job = eval_job_for_autoencoder_kmeans
    #     task_func = functools.partial(parallel_eval_job, train_episodes_raw=train_episodes_raw,
    #                                     test_episodes_raw=test_episodes_raw,
    #                                     all_caths_train=all_caths_train,
    #                                     all_caths_test=all_caths_test)
        
    # create paths for the models and the processed data
    os.makedirs(os.path.join(tc.models_path, f'{experiment_type}_models'), exist_ok=True)  # to save the clustering models
    os.makedirs(os.path.join(tc.models_path, f'{experiment_name}_rl_policies'), exist_ok=True)  # to save the policies (different per experiment)
    os.makedirs(os.path.join(tc.processed_data_path, 'train', f'{experiment_type}'), exist_ok=True)  # to save the clustering results
    os.makedirs(os.path.join(tc.processed_data_path, 'test', f'{experiment_type}'), exist_ok=True)  # to save the clustering results
    os.makedirs(os.path.join(tc.processed_data_path, 'validation', f'{experiment_type}'), exist_ok=True)  # to save the clustering results

    if experiment_type.startswith('autoencoder'):
        # encode the features using the autoencoder
        X_train = torch.tensor(all_caths_train_imputed.values).float().to(device)
        X_test = torch.tensor(all_caths_test_imputed.values).float().to(device)
        all_caths_train_imputed_encoded = autoencoder.encode(X_train).detach().cpu().numpy()
        all_caths_test_imputed_encoded = autoencoder.encode(X_test).detach().cpu().numpy()
        all_caths_train_imputed = pd.DataFrame(all_caths_train_imputed_encoded, columns=[f'feature_{i}' for i in range(all_caths_train_imputed_encoded.shape[1])])
        all_caths_test_imputed = pd.DataFrame(all_caths_test_imputed_encoded, columns=[f'feature_{i}' for i in range(all_caths_test_imputed_encoded.shape[1])])

        print(f"Finished encoding the features using the autoencoder - latent dim = {params['latent_dim']}")

    parallel_eval_job = eval_job_for_autoencoder_kmeans
    task_func = functools.partial(parallel_eval_job, train_episodes_raw=train_episodes_raw,
                                    test_episodes_raw=test_episodes_raw,
                                    all_caths_train=all_caths_train,
                                    all_caths_test=all_caths_test,
                                    all_caths_train_imputed=all_caths_train_imputed,
                                    all_caths_test_imputed=all_caths_test_imputed,
                                    reward_function=reward_function,
                                    read_clusters_from_file=True,
                                    experiment_type=experiment_type)


    n_jobs = max(1, os.cpu_count() - 7)  # use all available CPUs except 7  
    all_results = Parallel(n_jobs=n_jobs)(delayed(task_func)(n_clusters) for n_clusters in n_clusters_list)

    # Convert all results to a DataFrame
    results_df = pd.DataFrame(all_results)

    # Evaluate the policies with the behavior policy: Behavior Policy itself, a Random Policy, and Single Action Policies (CABG, Medical Therapy, PCI)
    behavior_plicy = PolicyResolver('behavior_physician_policy', list(actions_dict.values()))
    for dataset_name, dataset in [('test', test_episodes_raw), ('train', train_episodes_raw)]:
        for policy_name in ['behavior_physician_policy', 'random'] + list(actions_dict.keys()):
            policy_resolver = PolicyResolver(policy_name, list(actions_dict.values()))
            wis, ci = weighted_importance_sampling_with_bootstrap(dataset, 0.99, policy_resolver, behavior_plicy,
                                                                  num_bootstrap_samples=1000, N=1000, confidedence_level=0.95)
            results_df[f'{policy_name}_{dataset_name}'] = wis
            results_df[f'{policy_name}_{dataset_name}_LB'] = ci[0]
            results_df[f'{policy_name}_{dataset_name}_UB'] = ci[1]

    # Save the results to a CSV file
    file_name = os.path.join(tc.EXPERIMENTS_RESULTS, f'{experiment_name}.csv')
    results_df.sort_values(by='n_clusters', inplace=True)
    results_df.to_csv(file_name, index=False)