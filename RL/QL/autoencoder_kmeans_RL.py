import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sklearn as sk
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import training_constants as tc
from rl_utils import Episode, Transition, get_episodes, reward_func_mace, ClusteringBasedInference
import datetime
from torch.optim import Adam
import mdptoolbox
from rl_utils import PolicyResolver, weighted_importance_sampling_with_bootstrap
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from train_physician_imitation_model import PhysicianImitator
import torch
import functools
from autoencoder_training import Autoencoder
from training_pipeline3 import run_kmeans_and_save
import joblib
from joblib import Parallel, delayed
import argparse

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        

def do_q_learning_and_evaluate_policy(n_clusters,
                                      all_caths_train, all_caths_test,
                                      all_caths_train_clusters, all_caths_test_clusters,
                                      train_episodes_raw, test_episodes_raw):

    # Load the data
    behavior_plicy_name = 'physician_imitator'
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
    treatments = tc.treatments
    action_encoder = preprocessing.LabelEncoder()
    action_encoder.fit(np.array(treatments).reshape(-1, 1))
    actions_dict = dict(zip(action_encoder.classes_, action_encoder.transform(action_encoder.classes_)))
    # print("Actions dictionary:", actions_dict)

    # Create the episodes
    train_episodes = get_episodes(all_caths_train, all_caths_train_clusters, action_encoder, rewards_list, reward_func_mace)
    test_episodes = get_episodes(all_caths_test, all_caths_test_clusters, action_encoder, rewards_list, reward_func_mace)


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

    # calculate softmax of Q-matrix
    optimal_policy_probs = np.zeros_like(Q_matrix)
    for i in range(Q_matrix.shape[0]):
        max_q = np.max(Q_matrix[i, :])  # Find the maximum value in the Q-matrix row to avoid numerical instability (overflow)
        optimal_policy_probs[i, :] = np.exp(Q_matrix[i, :] - max_q) / np.sum(np.exp(Q_matrix[i, :] - max_q))

    # calculate the optimal policy and print them
    # for state in range(n_clusters):
    #     op = optimal_policy[state]
    #     pp = np.argmax(physician_policy[state, :])
        # print(f'State {state}: physician policy = {list(actions_dict.keys())[pp]}, optimal policy = {list(actions_dict.keys())[op]}')

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


    # resolve the policies
    physician_policy_resolver = PolicyResolver(physician_policy, list(actions_dict.values()))
    greedy_physician_policy_resolver = PolicyResolver(greedy_physician_policy, list(actions_dict.values()))
    optimal_policy_resolver = PolicyResolver(optimal_policy_probs, list(actions_dict.values()))
    greedy_optimal_policy_resolver = PolicyResolver(greedy_optimal_policy, list(actions_dict.values()))

    # Evaluate the policies
    results = {}
    # print(f"Evaluating the policies using weighted importance sampling with bootstrap for number of clusters = {n_clusters}")
    wis, ci = weighted_importance_sampling_with_bootstrap(test_episodes, 0.99, physician_policy_resolver, physician_policy_resolver, num_bootstrap_samples=500, N=1000, confidedence_level=0.95)
    results['physician'] = [wis, ci[0], ci[1]]
    wis, ci = weighted_importance_sampling_with_bootstrap(train_episodes, 0.99, physician_policy_resolver, physician_policy_resolver, num_bootstrap_samples=500, N=1000, confidedence_level=0.95)
    results['physician_train'] = [wis, ci[0], ci[1]]
    # print(f"Weighted importance sampling for the physician's policy: {wis}, CI: {ci}")
    wis, ci = weighted_importance_sampling_with_bootstrap(test_episodes, 0.99, greedy_physician_policy_resolver, physician_policy_resolver, num_bootstrap_samples=500, N=1000, confidedence_level=0.95)
    results['greedy_physician'] = [wis, ci[0], ci[1]]
    wis, ci = weighted_importance_sampling_with_bootstrap(train_episodes, 0.99, greedy_physician_policy_resolver, physician_policy_resolver, num_bootstrap_samples=500, N=1000, confidedence_level=0.95)
    results['greedy_physician_train'] = [wis, ci[0], ci[1]]   
    # print(f"Weighted importance sampling for the greedy physician's policy: {wis}, CI: {ci}")
    wis, ci = weighted_importance_sampling_with_bootstrap(test_episodes, 0.99, optimal_policy_resolver, physician_policy_resolver, num_bootstrap_samples=500, N=1000, confidedence_level=0.95)
    results['optimal'] = [wis, ci[0], ci[1]]
    wis, ci = weighted_importance_sampling_with_bootstrap(train_episodes, 0.99, optimal_policy_resolver, physician_policy_resolver, num_bootstrap_samples=500, N=1000, confidedence_level=0.95)
    results['optimal_train'] = [wis, ci[0], ci[1]]
    # print(f"Weighted importance sampling for the optimal policy: {wis}, CI: {ci}")
    wis, ci = weighted_importance_sampling_with_bootstrap(test_episodes, 0.99, greedy_optimal_policy_resolver, physician_policy_resolver, num_bootstrap_samples=500, N=1000, confidedence_level=0.95)
    results['greedy_optimal'] = [wis, ci[0], ci[1]]
    # print('---------------------------------------------------------------------------')

    # put the optimal policy inside the test episodes
    for ep, episode in enumerate(test_episodes):
        raw_episode = test_episodes_raw[ep]
        for tr, transition in enumerate(episode):
            state = transition.state
            state = int(state[0])
            transition.prediction_probs['optimal'] = optimal_policy_probs[state, :] # put the optimal policy inside the episode
            transition.prediction_probs[behavior_plicy_name] = raw_episode.transitions[tr].prediction_probs[behavior_plicy_name] # put the physician's policy inside the episode

    # put the optimal policy inside the train episodes
    for ep, episode in enumerate(train_episodes):
        raw_episode = train_episodes_raw[ep]
        for tr, transition in enumerate(episode):
            state = transition.state
            state = int(state[0])
            transition.prediction_probs['optimal'] = optimal_policy_probs[state, :] # put the optimal policy inside the episode
            transition.prediction_probs[behavior_plicy_name] = raw_episode.transitions[tr].prediction_probs[behavior_plicy_name] # put the physician's policy inside the episode

    # # evaluate using the imitator model as the behavior policy
    behavior_plicy = PolicyResolver(behavior_plicy_name, list(actions_dict.values()))
    clustering_policy = PolicyResolver('optimal', list(actions_dict.values()))
    wis, ci = weighted_importance_sampling_with_bootstrap(test_episodes, 0.99, clustering_policy, behavior_plicy, num_bootstrap_samples=500, N=1000, confidedence_level=0.95)
    results['optimal_vs_imitation'] = [wis, ci[0], ci[1]]
    wis, ci = weighted_importance_sampling_with_bootstrap(train_episodes, 0.99, clustering_policy, behavior_plicy, num_bootstrap_samples=500, N=1000, confidedence_level=0.95)
    results['optimal_vs_imitation_train'] = [wis, ci[0], ci[1]]

    return results, policies

def eval_job_for_vanilla_kmeans(n_clusters,
                      all_caths_train, all_caths_test,
                      train_episodes_raw, test_episodes_raw):
    """
    Evaluate the policy for a specific number of clusters

    """
    all_caths_train_clusters = pd.read_csv(os.path.join(processed_data_path, 'train', f'cath_kmeans_clusters_{n_clusters}.csv'))
    all_caths_test_clusters = pd.read_csv(os.path.join(processed_data_path, 'test', f'cath_kmeans_clusters_{n_clusters}.csv'))
    
    results,policies = do_q_learning_and_evaluate_policy(n_clusters,
                                                all_caths_train, all_caths_test,
                                                all_caths_train_clusters, all_caths_test_clusters,
                                                train_episodes_raw, test_episodes_raw)
    results_row = {'n_clusters': n_clusters,
                    'physician_wis': results['physician'][0],
                    'physician_LB': results['physician'][1],
                    'physician_UB': results['physician'][2],
                    'greedy_physician_wis': results['greedy_physician'][0],
                    'greedy_physician_LB': results['greedy_physician'][1],
                    'greedy_physician_UB': results['greedy_physician'][2],
                    'optimal_wis': results['optimal'][0],
                    'optimal_LB': results['optimal'][1],
                    'optimal_UB': results['optimal'][2],
                    'optimal_vs_imitation_wis': results['optimal_vs_imitation'][0],
                    'optimal_vs_imitation_LB': results['optimal_vs_imitation'][1],
                    'optimal_vs_imitation_UB': results['optimal_vs_imitation'][2],
                    'physician_train_wis': results['physician_train'][0],
                    'physician_train_LB': results['physician_train'][1],
                    'physician_train_UB': results['physician_train'][2],
                    'greedy_physician_train_wis': results['greedy_physician_train'][0],
                    'greedy_physician_train_LB': results['greedy_physician_train'][1],
                    'greedy_physician_train_UB': results['greedy_physician_train'][2],
                    'optimal_train_wis': results['optimal_train'][0],
                    'optimal_train_LB': results['optimal_train'][1],
                    'optimal_train_UB': results['optimal_train'][2],
                    'greedy_optimal_wis': results['greedy_optimal'][0],
                    'greedy_optimal_LB': results['greedy_optimal'][1],
                    'greedy_optimal_UB': results['greedy_optimal'][2],
                    'optimal_vs_imitation_train_wis': results['optimal_vs_imitation_train'][0],
                    'optimal_vs_imitation_train_LB': results['optimal_vs_imitation_train'][1],
                    'optimal_vs_imitation_train_UB': results['optimal_vs_imitation_train'][2]
                    }
    
    print(f"Finished evaluating the policies for number of clusters = {n_clusters}")

    return results_row


def eval_job_for_autoencoder_kmeans(n_clusters,
                      all_caths_train, all_caths_test,
                      train_episodes_raw, test_episodes_raw,
                      all_caths_train_imputed, all_caths_test_imputed,
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


    results,policies = do_q_learning_and_evaluate_policy(n_clusters,
                                                all_caths_train, all_caths_test,
                                                all_caths_train_clusters, all_caths_test_clusters,
                                                train_episodes_raw, test_episodes_raw)
    
    policies_file = os.path.join(tc.models_path, f'{experiment_type}_models', f'policies_{n_clusters}.pkl')
    pickle.dump(policies, open(policies_file, 'wb'))
    
    results_row = {'n_clusters': n_clusters,
                    'physician_wis': results['physician'][0],
                    'physician_LB': results['physician'][1],
                    'physician_UB': results['physician'][2],
                    'greedy_physician_wis': results['greedy_physician'][0],
                    'greedy_physician_LB': results['greedy_physician'][1],
                    'greedy_physician_UB': results['greedy_physician'][2],
                    'optimal_wis': results['optimal'][0],
                    'optimal_LB': results['optimal'][1],
                    'optimal_UB': results['optimal'][2],
                    'optimal_vs_imitation_wis': results['optimal_vs_imitation'][0],
                    'optimal_vs_imitation_LB': results['optimal_vs_imitation'][1],
                    'optimal_vs_imitation_UB': results['optimal_vs_imitation'][2],
                    'physician_train_wis': results['physician_train'][0],
                    'physician_train_LB': results['physician_train'][1],
                    'physician_train_UB': results['physician_train'][2],
                    'greedy_physician_train_wis': results['greedy_physician_train'][0],
                    'greedy_physician_train_LB': results['greedy_physician_train'][1],
                    'greedy_physician_train_UB': results['greedy_physician_train'][2],
                    'optimal_train_wis': results['optimal_train'][0],
                    'optimal_train_LB': results['optimal_train'][1],
                    'optimal_train_UB': results['optimal_train'][2],
                    'greedy_optimal_wis': results['greedy_optimal'][0],
                    'greedy_optimal_LB': results['greedy_optimal'][1],
                    'greedy_optimal_UB': results['greedy_optimal'][2],
                    'optimal_vs_imitation_train_wis': results['optimal_vs_imitation_train'][0],
                    'optimal_vs_imitation_train_LB': results['optimal_vs_imitation_train'][1],
                    'optimal_vs_imitation_train_UB': results['optimal_vs_imitation_train'][2]
                    }
    
    print(f"Finished evaluating the policies for number of clusters = {n_clusters}")
    return results_row


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the clustering-based inference using Q-Learning')
    parser.add_argument('--experiment_type', type=str, default='autoencoder_sigmoid_8_kmeans', help='The type of the experiment (autoencoder_kmeans, autoencoder_sigmoid_8_kmeans, vanilla_kmeans)')
    parser.add_argument('--autoencoder_name', type=str, default='autoencoder_sigmoid_8', help='The name of the autoencoder model')

    args = parser.parse_args()
    experiment_type = args.experiment_type
    autoencoder_name = args.autoencoder_name

    n_clusters_list = tc.n_clusters_list_cath
    # n_clusters_list = range(10, 120)
    # experiment_type = 'autoencoder_sigmoid_8_kmeans'
    # autoencoder_name = 'autoencoder_sigmoid_8'
    experiment_name = f'{experiment_type}-{n_clusters_list[0]}-{n_clusters_list[-1]}-all-caths' + '-reward-' + 'mace' + '__' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # autoencoder loading
    autoencoder_model_path = os.path.join(tc.models_path, autoencoder_name, 'checkpoint.pth')
    checkpoint = torch.load(autoencoder_model_path)
    params = checkpoint['parameters']
    autoencoder = Autoencoder(params['input_dim'], params['latent_dim'], params['num_hidden_layers'], params['activation_fn'])
    autoencoder.load_state_dict(checkpoint['state_dict'])
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

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
    train_episodes_raw = get_episodes(all_caths_train, all_caths_train_imputed, action_encoder, rewards_list, reward_func_mace)
    test_episodes_raw = get_episodes(all_caths_test, all_caths_test_imputed, action_encoder, rewards_list, reward_func_mace)


    # put the behavior plicy of the imitator model (logistic regression) inside the episodes
    model_path = os.path.join(models_path, 'physician_imitation', 'physician_imitation_logistic_model.pkl')
    physician_imitator = pickle.load(open(model_path, 'rb'))
    for dataset in [train_episodes_raw, test_episodes_raw]:
        for episode in dataset:
            for transition in episode:
                state = transition.state
                sample = state.astype(np.float32).reshape(1, -1)
                logits = physician_imitator.predict_proba(sample)
                transition.prediction_probs['physician_imitator'] = logits.reshape(-1)  # imitator model's policy
                
                # random action policies
                random_action = np.random.choice(list(actions_dict.values()))
                transition.prediction_probs['random'] = np.zeros(len(actions_dict))
                transition.prediction_probs['random'][random_action] = 1.0

                # single action policies
                for action_name, action_id in actions_dict.items():
                    transition.prediction_probs[action_name] = np.zeros(len(actions_dict))
                    transition.prediction_probs[action_name][action_id] = 1.0

                


    if experiment_type == 'vanilla_kmeans':
        parallel_eval_job = eval_job_for_vanilla_kmeans
        task_func = functools.partial(parallel_eval_job, train_episodes_raw=train_episodes_raw,
                                        test_episodes_raw=test_episodes_raw,
                                        all_caths_train=all_caths_train,
                                        all_caths_test=all_caths_test)
        
    elif experiment_type.startswith('autoencoder'):
        # create paths for the models and the processed data
        os.makedirs(os.path.join(tc.models_path, f'{experiment_type}_models'), exist_ok=True)
        os.makedirs(os.path.join(tc.processed_data_path, 'train', f'{experiment_type}'), exist_ok=True)
        os.makedirs(os.path.join(tc.processed_data_path, 'test', f'{experiment_type}'), exist_ok=True)
        os.makedirs(os.path.join(tc.processed_data_path, 'validation', f'{experiment_type}'), exist_ok=True)

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
                                        read_clusters_from_file=True,
                                        experiment_type=experiment_type)

    # # Process the data
    # all_results = []
    # with ProcessPoolExecutor(max_workers=30) as executor:
    #     # Schedule the tasks
    #     futures = [executor.submit(task_func, n_clusters) for n_clusters in n_clusters_list]
        
    #     for future in as_completed(futures):
    #         all_results.append(future.result())
    n_jobs = max(1, os.cpu_count() - 7)   
    all_results = Parallel(n_jobs=n_jobs)(delayed(task_func)(n_clusters) for n_clusters in n_clusters_list)
    # for n_clusters in n_clusters_list:
    #     all_results.append(task_func(n_clusters))

    #     results_df_temp = pd.DataFrame(all_results)
    #     results_df_temp.to_csv(os.path.join('EXPERIMENTS_RESULTS', f'{experiment_name}_temp.csv'), index=False)

    # Convert all results to a DataFrame
    results_df = pd.DataFrame(all_results)

    # add the expected values of the behavior policy to the results
    behavior_plicy = PolicyResolver('physician_imitator', list(actions_dict.values()))
    wis, ci = weighted_importance_sampling_with_bootstrap(test_episodes_raw, 0.99, behavior_plicy, behavior_plicy, num_bootstrap_samples=500, N=1000, confidedence_level=0.95)
    results_df['physician_imitator_wis'] = wis
    results_df['physician_imitator_LB'] = ci[0]
    results_df['physician_imitator_UB'] = ci[1]
    wis, ci = weighted_importance_sampling_with_bootstrap(train_episodes_raw, 0.99, behavior_plicy, behavior_plicy, num_bootstrap_samples=500, N=1000, confidedence_level=0.95)
    results_df['physician_imitator_train_wis'] = wis
    results_df['physician_imitator_train_LB'] = ci[0]
    results_df['physician_imitator_train_UB'] = ci[1]

    # add the expected values of the random policy to the results
    random_policy = PolicyResolver('random', list(actions_dict.values()))
    wis, ci = weighted_importance_sampling_with_bootstrap(test_episodes_raw, 0.99, random_policy, behavior_plicy, num_bootstrap_samples=500, N=1000, confidedence_level=0.95)
    results_df['random_wis'] = wis
    results_df['random_LB'] = ci[0]
    results_df['random_UB'] = ci[1]
    wis, ci = weighted_importance_sampling_with_bootstrap(train_episodes_raw, 0.99, random_policy, behavior_plicy, num_bootstrap_samples=500, N=1000, confidedence_level=0.95)
    results_df['random_train_wis'] = wis
    results_df['random_train_LB'] = ci[0]
    results_df['random_train_UB'] = ci[1]

    # add the expected values of the single action policies to the results
    for action_name, action_id in actions_dict.items():
        single_action_policy = PolicyResolver(action_name, list(actions_dict.values()))
        wis, ci = weighted_importance_sampling_with_bootstrap(test_episodes_raw, 0.99, single_action_policy, behavior_plicy, num_bootstrap_samples=500, N=1000, confidedence_level=0.95)
        results_df[f'{action_name}_wis'] = wis
        results_df[f'{action_name}_LB'] = ci[0]
        results_df[f'{action_name}_UB'] = ci[1]
        wis, ci = weighted_importance_sampling_with_bootstrap(train_episodes_raw, 0.99, single_action_policy, behavior_plicy, num_bootstrap_samples=500, N=1000, confidedence_level=0.95)
        results_df[f'{action_name}_train_wis'] = wis
        results_df[f'{action_name}_train_LB'] = ci[0]
        results_df[f'{action_name}_train_UB'] = ci[1]


    # Save the results to a CSV file
    file_name = os.path.join('EXPERIMENTS_RESULTS', f'{experiment_name}.csv')
    results_df.sort_values(by='n_clusters', inplace=True)
    results_df.to_csv(file_name, index=False)