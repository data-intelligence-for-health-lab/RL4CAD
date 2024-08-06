import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sklearn as sk
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import training_constants as tc
from rl_utils import Episode, Transition, get_episodes, get_d3rlpy_dataset
from rl_utils import reward_func_mace, reward_func_mace_survival, reward_func_mace_survival_repvasc
import d3rlpy
from torchinfo import summary
import torch
import datetime
from torch.optim import Adam
import argparse
from rl_utils import PolicyResolver, weighted_importance_sampling_with_bootstrap
from train_physician_imitation_model import PhysicianImitator
from autoencoder_training import Autoencoder
import itertools

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def fit_dqn(train_dataset, algo, experiment_name, model_params):
    num_layers = model_params['num_layers']
    num_hidden_neurons = model_params['num_hidden_neurons']
    activation_func = model_params['activation_func']
    learning_rate = model_params['learning_rate']
    n_steps = model_params['n_steps']
    
    # encoder factory
    encoder_factory = d3rlpy.models.VectorEncoderFactory(
        hidden_units=[num_hidden_neurons] * num_layers,
        activation=activation_func,
    )

    # optimizer factory
    # optimizer_factory = d3rlpy.models.OptimizerFactory(Adam, lr=0.001)

    if algo == 'DQN':
        # set DQN config
        rl_model = d3rlpy.algos.DQNConfig(encoder_factory=encoder_factory, learning_rate=learning_rate).create(device='cuda:2')

    elif algo == 'DDQN':
        q_func = d3rlpy.models.QRQFunctionFactory(n_quantiles=32)
        rl_model = d3rlpy.algos.DQNConfig(q_func_factory=q_func, encoder_factory=encoder_factory, learning_rate=learning_rate).create(device='cuda:2')

    elif algo == 'CQL':
        conservative_alpha = model_params['conservative_alpha']
        rl_model = d3rlpy.algos.DiscreteCQLConfig(alpha=conservative_alpha, encoder_factory=encoder_factory, learning_rate=learning_rate).create(device='cuda:2')

    elif algo == 'DCQL':
        conservative_alpha = model_params['conservative_alpha']
        q_func = d3rlpy.models.QRQFunctionFactory(n_quantiles=32)
        rl_model = d3rlpy.algos.DiscreteCQLConfig(alpha=conservative_alpha, q_func_factory=q_func, encoder_factory=encoder_factory, learning_rate=learning_rate).create(device='cuda:2')
    else:
        raise ValueError("Invalid experiment name")
            
    # build the model
    rl_model.build_with_dataset(train_dataset)
    # td_error_evaluator = d3rlpy.metrics.TDErrorEvaluator(episodes=train_dataset.episodes)

    # train the RL model
    rl_model.fit(
        train_dataset,
        n_steps=n_steps,
        experiment_name= experiment_name)
    
    return rl_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a DRL model using d3rlpy')
    parser.add_argument('--algo', type=str, default='DQN', help='The algorithm to use for training the model')
    parser.add_argument('--reward', type=str, default='mace-survival-repvasc', help='The reward function to use for training the model')
    parser.add_argument('--experiment_name', type=str, default='', help='The name of the experiment')
    args = parser.parse_args()

    algo = args.algo

    reward_name = args.reward
    if reward_name == 'mace':
        reward_function = reward_func_mace
    elif reward_name == 'mace-survival':
        reward_function = reward_func_mace_survival
    elif reward_name == 'mace-survival-repvasc':
        reward_function = reward_func_mace_survival_repvasc

    # dataframe to store the results
    evaluation_results = pd.DataFrame(columns=['experiment_name', 'num_layers', 'num_hidden_neurons', 'activation_func', 'learning_rate', 'n_steps',
                                               'rl_policy_test_wis', 'rl_policy_test_ci',
                                               'rl_policy_validation_wis', 'rl_policy_validation_ci',
                                               'rl_policy_train_wis', 'rl_policy_train_ci',
                                               'rl_greedy_policy_test_wis', 'rl_greedy_policy_test_ci',
                                               'rl_greedy_policy_validation_wis', 'rl_greedy_policy_validation_ci',
                                                'rl_greedy_policy_train_wis', 'rl_greedy_policy_train_ci'])

    use_autoencoder = False
    autoencoder_n = 32
    autoencoder_type = 'autoencoder_sigmoid'
    if use_autoencoder:
        experiment_name = f"{algo}-with-{autoencoder_type}-{autoencoder_n}-all-caths-reward-{reward_name}__{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}" if args.experiment_name == '' else args.experiment_name
    else:
        experiment_name = f"{algo}-all-caths-reward-{reward_name}__{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}" if args.experiment_name == '' else args.experiment_name
    # experiment_name = 'DQN-all-caths' + '-reward-' + 'survival-mace' + '__' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # experiment_name = 'CQL-all-caths-obstructiveCAD' + '-reward-' + 'mace' + '__' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    # Load the data
    processed_data_path = tc.processed_data_path
    models_path = tc.models_path
    rewards_list = tc.rewards_list
    os.makedirs(os.path.join(models_path, experiment_name), exist_ok=True)

    all_caths_train = pd.read_csv(os.path.join(processed_data_path, 'train', 'all_caths.csv'))
    all_caths_train['SubsequentTreatment'].fillna('Medical Therapy', inplace=True)
    treatment_train = all_caths_train['SubsequentTreatment']
    all_caths_train_imputed = pd.read_csv(os.path.join(processed_data_path, 'train', 'all_caths_train_imputed.csv'))

    all_caths_test = pd.read_csv(os.path.join(processed_data_path, 'test', 'all_caths.csv'))
    all_caths_test['SubsequentTreatment'].fillna('Medical Therapy', inplace=True)
    treatment_test = all_caths_test['SubsequentTreatment']
    all_caths_test_imputed = pd.read_csv(os.path.join(processed_data_path, 'test', 'all_caths_test_imputed.csv'))

    all_caths_validation = pd.read_csv(os.path.join(processed_data_path, 'validation', 'all_caths.csv'))
    all_caths_validation['SubsequentTreatment'].fillna('Medical Therapy', inplace=True)
    treatment_validation = all_caths_validation['SubsequentTreatment']
    all_caths_validation_imputed = pd.read_csv(os.path.join(processed_data_path, 'validation', 'all_caths_validation_imputed.csv'))

    # use the latent layer of an autoencoder instead of the raw data
    if use_autoencoder:
        autoencoder_name = f'{autoencoder_type}_{autoencoder_n}'
        autoencoder_path = os.path.join(models_path, autoencoder_name, 'checkpoint.pth')
        autoencoder_checkpoint = torch.load(autoencoder_path)
        params = autoencoder_checkpoint['parameters']
        autoencoder = Autoencoder(params['input_dim'], params['latent_dim'], params['num_hidden_layers'], params['activation_fn'])
        autoencoder.load_state_dict(autoencoder_checkpoint['state_dict'])
        autoencoder = autoencoder.to(device)
        autoencoder.eval()
        print(f"Loaded the autoencoder model from {autoencoder_path}")
        
        # encode the features using the autoencoder
        X_train = torch.tensor(all_caths_train_imputed.values).float().to(device)
        X_test = torch.tensor(all_caths_test_imputed.values).float().to(device)
        all_caths_train_imputed_encoded = autoencoder.encode(X_train).detach().cpu().numpy()
        all_caths_test_imputed_encoded = autoencoder.encode(X_test).detach().cpu().numpy()
        all_caths_train_imputed = pd.DataFrame(all_caths_train_imputed_encoded, columns=[f'feature_{i}' for i in range(all_caths_train_imputed_encoded.shape[1])])
        all_caths_test_imputed = pd.DataFrame(all_caths_test_imputed_encoded, columns=[f'feature_{i}' for i in range(all_caths_test_imputed_encoded.shape[1])])
        print(f"Encoded the features using the autoencoder with {autoencoder_n} latent dimensions")
    

    # encode the actions
    treatments = ['Medical Therapy', 'PCI', 'CABG']
    action_encoder = preprocessing.LabelEncoder()
    action_encoder.fit(np.array(treatments).reshape(-1, 1))
    actions_dict = dict(zip(action_encoder.classes_, action_encoder.transform(action_encoder.classes_)))
    print("Actions dictionary:", actions_dict)

    # Create the episodes
    train_episodes = get_episodes(all_caths_train, all_caths_train_imputed, action_encoder, rewards_list, reward_function)
    test_episodes = get_episodes(all_caths_test, all_caths_test_imputed, action_encoder, rewards_list, reward_function)
    validation_episodes = get_episodes(all_caths_validation, all_caths_validation_imputed, action_encoder, rewards_list, reward_function)

    # create the d3rlpy datasets
    train_dataset = get_d3rlpy_dataset(train_episodes, actions_dict)
    # test_dataset = get_d3rlpy_dataset(test_episodes, actions_dict)
    # validation_dataset = get_d3rlpy_dataset(validation_episodes, actions_dict)


    # behavior policy
    best_n_clusters = tc.n_clusters_behavior_policy
    best_physician_experiment_type = tc.experiment_type_behavior_policy
    best_physician_policy = pickle.load(open(tc.behavior_policy_path, 'rb'))
    best_physician_policy = best_physician_policy['physician']
    train_clusters_path = os.path.join(tc.processed_data_path, 'train', f'{best_physician_experiment_type}', f'{best_physician_experiment_type}_kmeans_clusters_{best_n_clusters}.csv')
    test_clusters_path = os.path.join(tc.processed_data_path, 'test', f'{best_physician_experiment_type}', f'{best_physician_experiment_type}_kmeans_clusters_{best_n_clusters}.csv')
    validation_clusters_path = os.path.join(tc.processed_data_path, 'validation', f'{best_physician_experiment_type}', f'{best_physician_experiment_type}_kmeans_clusters_{best_n_clusters}.csv')
    all_caths_train_clusters = pd.read_csv(train_clusters_path)
    all_caths_test_clusters = pd.read_csv(test_clusters_path)
    all_caths_validation_clusters = pd.read_csv(validation_clusters_path)

    behavior_train_episodes = get_episodes(all_caths_train, all_caths_train_clusters, action_encoder, rewards_list, reward_function)
    behavior_test_episodes = get_episodes(all_caths_test, all_caths_test_clusters, action_encoder, rewards_list, reward_function)
    behavior_validation_episodes = get_episodes(all_caths_validation, all_caths_validation_clusters, action_encoder, rewards_list, reward_function)

    # put the behavior policy inside the reference episodes
    for dataset, behavior_dataset in [(train_episodes, behavior_train_episodes), (test_episodes, behavior_test_episodes), (validation_episodes, behavior_validation_episodes)]:
        for episode, behavior_episode in zip(dataset, behavior_dataset):
            for transition, behavior_transition in zip(episode, behavior_episode):
                behavior_state = behavior_transition.state
                behavior_state = int(behavior_state[0])
                transition.prediction_probs['best_physician_policy'] = best_physician_policy[behavior_state, :]

                # single action policies
                for action_name, action in actions_dict.items():
                    transition.prediction_probs[f'single_action_policy_{action_name}'] = np.zeros(len(actions_dict))
                    transition.prediction_probs[f'single_action_policy_{action_name}'][action] = 1.0


    # encoder factory
    # num_layers_list = [1, 2, 3, 4]
    # num_hidden_neurons_list = [8, 16, 32, 64]
    # activation_func_list = ['relu', 'leaky_relu', 'tanh']
    # learning_rate_list = [0.01, 0.001, 0.0001]
    # n_steps_list = [100000]
    num_layers_list = [2]
    num_hidden_neurons_list = [8]
    activation_func_list = ['relu']
    learning_rate_list = [0.01]
    n_steps_list = [2000000]

    if algo == 'CQL' or algo == 'DCQL':
        conservative_alpha_list = [0.001, 0.01, 0.1, 0.5]
    else:
        conservative_alpha_list = [None]

    for num_layers, num_hidden_neurons, activation_func, learning_rate, n_steps, conservative_alpha in itertools.product(num_layers_list,
                                                                                                     num_hidden_neurons_list,
                                                                                                     activation_func_list,
                                                                                                     learning_rate_list,
                                                                                                     n_steps_list,
                                                                                                     conservative_alpha_list):
        model_params = {
            'num_layers': num_layers,
            'num_hidden_neurons': num_hidden_neurons,
            'activation_func': activation_func,
            'learning_rate': learning_rate,
            'n_steps': n_steps,
        }
        # for CQL
        if conservative_alpha is not None:
            model_params['conservative_alpha'] = conservative_alpha


        # fit the model
        rl_model = fit_dqn(train_dataset, algo, experiment_name, model_params)

        # save full parameters and configurations in a single file.
        if conservative_alpha is None:
            # for normal DQN
            model_name = f"{num_layers}__{num_hidden_neurons}__{activation_func}__{learning_rate}__{n_steps}"
        else:
            # for CQL
            model_name = f"{num_layers}__{num_hidden_neurons}__{activation_func}__{learning_rate}__{n_steps}__{conservative_alpha}"

        model_path = os.path.join(models_path, experiment_name, model_name + '.d3')

        # # TODO: remove this line
        # model_import_folder = 'models_obstructive_cad5/DQN-all-caths-reward-mace__20240415-155033'
        # model_import_path = os.path.join(model_import_folder, model_name + '.d3')
        # rl_model = d3rlpy.load_learnable(model_import_path)
        
        rl_model.save(model_path)
        print('Model saved at:', model_path)


        # evaluate the model
        rl_policy = PolicyResolver(rl_model, list(actions_dict.values()))
        rl_greedy_policy = PolicyResolver(rl_model, list(actions_dict.values()), greedy=True)
        physician_policy = PolicyResolver('best_physician_policy', list(actions_dict.values()))
        physician_policy_greedy = PolicyResolver('best_physician_policy', list(actions_dict.values()), greedy=True)

        policy_eval_results = {}
        policy_eval_results['experiment_name'] = experiment_name
        for param_name, param_value in model_params.items():
            policy_eval_results[param_name] = param_value
        
        # evaluate the policies
        for eval_episodes, eval_name in zip([test_episodes, validation_episodes, train_episodes], ['test', 'validation', 'train']):
            for policy_name, policy in zip(['rl_policy', 'rl_greedy_policy'], [rl_policy, rl_greedy_policy, physician_policy, physician_policy_greedy]):
                wis, ci = weighted_importance_sampling_with_bootstrap(eval_episodes, 0.99, policy, physician_policy, num_bootstrap_samples=1000, N=1000)
                policy_eval_results[f'{policy_name}_{eval_name}_wis'] = wis
                policy_eval_results[f'{policy_name}_{eval_name}_ci'] = str(ci)

                print(model_params)
                print(f"Weighted importance sampling for the {policy_name} on the {eval_name} data: {wis}, CI: {ci}")

        evaluation_results = pd.concat([evaluation_results, pd.DataFrame(policy_eval_results, index=[0])] , ignore_index=True)
        evaluation_results.to_csv(os.path.join(tc.EXPERIMENTS_RESULTS, f"{experiment_name}_evaluation_results_TEMP.csv"), index=False)

    
    # evaluate the best physician policy
    for eval_episodes, eval_name in zip([test_episodes, validation_episodes, train_episodes], ['test', 'validation', 'train']):
        wis, ci = weighted_importance_sampling_with_bootstrap(eval_episodes, 0.99, physician_policy, physician_policy, num_bootstrap_samples=1000, N=1000)
        evaluation_results[f'best_physician_policy_{eval_name}_wis'] = wis
        evaluation_results[f'best_physician_policy_{eval_name}_ci'] = str(ci)
        print(f"Weighted importance sampling for the best physician policy on the {eval_name} data: {wis}, CI: {ci}")

        # evaluate the greedy best physician policy
        wis, ci = weighted_importance_sampling_with_bootstrap(eval_episodes, 0.99, physician_policy_greedy, physician_policy, num_bootstrap_samples=1000, N=1000)
        evaluation_results[f'best_physician_policy_greedy_{eval_name}_wis'] = wis
        evaluation_results[f'best_physician_policy_greedy_{eval_name}_ci'] = str(ci)
        print(f"Weighted importance sampling for the greedy best physician policy on the {eval_name} data: {wis}, CI: {ci}")

        # evaluate the single action policies
        for action_name, action in actions_dict.items():
            single_action_policy = PolicyResolver(f'single_action_policy_{action_name}', list(actions_dict.values()))
            wis, ci = weighted_importance_sampling_with_bootstrap(eval_episodes, 0.99, single_action_policy, physician_policy, num_bootstrap_samples=1000, N=1000)
            evaluation_results[f'single_action_policy_{action_name}_{eval_name}_wis'] = wis
            evaluation_results[f'single_action_policy_{action_name}_{eval_name}_ci'] = str(ci)
            print(f"Weighted importance sampling for the single action policy {action_name} on the {eval_name} data: {wis}, CI: {ci}")


    evaluation_results.to_csv(os.path.join(tc.EXPERIMENTS_RESULTS, f"{experiment_name}_evaluation_results.csv"), index=False)


