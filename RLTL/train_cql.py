import d3rlpy
import pandas as pd
import os
import numpy as np
import pickle
from datetime import datetime
from sklearn import preprocessing
import itertools
from joblib import Parallel, delayed
import argparse
import transfer_constants as trc
from rl_utils import get_episodes, get_d3rlpy_dataset, PolicyResolver, weighted_importance_sampling_with_bootstrap, reward_func_mace_survival_repvasc

RESULTS_DF_COLS = ['experiment_name', 'num_layers', 'num_hidden_neurons', 'activation_func', 'learning_rate', 'n_steps',
                                               'rl_policy_test_wis', 'rl_policy_test_ci',
                                               'rl_policy_validation_wis', 'rl_policy_validation_ci',
                                               'rl_policy_train_wis', 'rl_policy_train_ci',
                                               'rl_greedy_policy_test_wis', 'rl_greedy_policy_test_ci',
                                               'rl_greedy_policy_validation_wis', 'rl_greedy_policy_validation_ci',
                                                'rl_greedy_policy_train_wis', 'rl_greedy_policy_train_ci']


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
        rl_model = d3rlpy.algos.DQNConfig(encoder_factory=encoder_factory, learning_rate=learning_rate).create(device='cuda')

    elif algo == 'DDQN':
        q_func = d3rlpy.models.QRQFunctionFactory(n_quantiles=32)
        rl_model = d3rlpy.algos.DQNConfig(q_func_factory=q_func, encoder_factory=encoder_factory, learning_rate=learning_rate).create(device='cuda')

    elif algo == 'CQL':
        conservative_alpha = model_params['conservative_alpha']
        rl_model = d3rlpy.algos.DiscreteCQLConfig(alpha=conservative_alpha, encoder_factory=encoder_factory, learning_rate=learning_rate).create(device='cuda')

    elif algo == 'DCQL':
        conservative_alpha = model_params['conservative_alpha']
        q_func = d3rlpy.models.QRQFunctionFactory(n_quantiles=32)
        rl_model = d3rlpy.algos.DiscreteCQLConfig(alpha=conservative_alpha, q_func_factory=q_func, encoder_factory=encoder_factory, learning_rate=learning_rate).create(device='cuda')
    else:
        raise ValueError("Invalid experiment name")
            
    # build the model
    rl_model.build_with_dataset(train_dataset)
    # td_error_evaluator = d3rlpy.metrics.TDErrorEvaluator(episodes=train_dataset.episodes)

    # train the RL model
    rl_model.fit(
        train_dataset,
        n_steps=n_steps,
        experiment_name= experiment_name + str(datetime.now()),
        save_interval=1e100, show_progress=False)
    
    return rl_model

def extract_behavior_policy_from_episodes(episodes, actions_dict, states_list):
    n_actions = len(actions_dict)
    n_states = len(states_list)
    transition_matrix = np.zeros((n_states, n_actions))
    for episode in episodes:
        for transition in episode:
            state = int(transition.state[0])
            action = transition.action
            transition_matrix[state, action] += 1

    # calculate physician's policy (probability of each action for each state)
    sum_over_actions = np.sum(transition_matrix, axis=1)
    behavior_policy = np.divide(transition_matrix, sum_over_actions[:, None], out=np.zeros_like(transition_matrix), where=sum_over_actions[:, None] != 0)
    behavior_policy[sum_over_actions == 0, :] = 1 / n_actions

    return behavior_policy



def run_training_session(data_folder, algo, model_params, models_folder, reward_function, stratify_on, experiment_name):
    rewards_list = trc.rewards_list

    # dataframe to store the results
    evaluation_results_1 = pd.DataFrame(columns=RESULTS_DF_COLS)

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
    print("Actions dictionary:", actions_dict)

    # Create the episodes
    train_episodes = get_episodes(all_caths_train, all_caths_train_imputed, action_encoder, rewards_list, reward_function)
    test_episodes = get_episodes(all_caths_test, all_caths_test_imputed, action_encoder, rewards_list, reward_function)
    validation_episodes = get_episodes(all_caths_validation, all_caths_validation_imputed, action_encoder, rewards_list, reward_function)

    # create the dataset
    train_dataset = get_d3rlpy_dataset(train_episodes, actions_dict)
    # test_dataset = get_d3rlpy_dataset(test_episodes, actions_dict)
    # validation_dataset = get_d3rlpy_dataset(validation_episodes, actions_dict)

    # load the behavior policy and create the behavior episodes
    behavior_experiment_type = trc.experiment_type_behavior_policy
    behavior_physician_policy_path = os.path.join(models_folder, f'behavior_policy_{behavior_experiment_type}.pkl')
    behavior_physician_policy_data = pickle.load(open(behavior_physician_policy_path, 'rb'))
    behavior_policy = behavior_physician_policy_data['behavior_policy']
    behavior_n_clusters = behavior_physician_policy_data['n_clusters']
    behavior_model_path = os.path.join(models_folder, f"{behavior_experiment_type}_models", f"{behavior_experiment_type}_{behavior_n_clusters}.pkl")
    behavior_policy_cluster_model = pickle.load(open(behavior_model_path, 'rb'))  # read the model to get the clusters
    behavior_train_clusters = behavior_policy_cluster_model.predict(all_caths_train_imputed)
    behavior_train_clusters = pd.DataFrame(behavior_train_clusters, columns=['cluster'])
    behavior_test_clusters = behavior_policy_cluster_model.predict(all_caths_test_imputed)
    behavior_test_clusters = pd.DataFrame(behavior_test_clusters, columns=['cluster'])
    behavior_validation_clusters = behavior_policy_cluster_model.predict(all_caths_validation_imputed)
    behavior_validation_clusters = pd.DataFrame(behavior_validation_clusters, columns=['cluster'])


    behavior_train_episodes = get_episodes(all_caths_train, behavior_train_clusters, action_encoder, rewards_list, reward_function)
    behavior_test_episodes = get_episodes(all_caths_test, behavior_test_clusters, action_encoder, rewards_list, reward_function)
    behavior_validation_episodes = get_episodes(all_caths_validation, behavior_validation_clusters, action_encoder, rewards_list, reward_function)

    # # extract the behavior policy
    # states_list = list(range(behavior_n_clusters + 1))
    # behavior_policy = extract_behavior_policy_from_episodes(behavior_train_episodes, actions_dict, states_list)

    # save the behavior policy
    behavior_policy_path = os.path.join(models_folder, experiment_name, 'behavior_policy.npy')
    os.makedirs(os.path.join(models_folder, experiment_name), exist_ok=True)
    np.save(behavior_policy_path, behavior_policy)


    # put the behavior policy inside the reference episodes
    for dataset, behavior_dataset in [(train_episodes, behavior_train_episodes), (test_episodes, behavior_test_episodes), (validation_episodes, behavior_validation_episodes)]:
        for episode, behavior_episode in zip(dataset, behavior_dataset):
            for transition, behavior_transition in zip(episode, behavior_episode):
                behavior_state = behavior_transition.state
                behavior_state = int(behavior_state[0])
                transition.prediction_probs['behavior_physician_policy'] = behavior_policy[behavior_state, :]

                # single action policies
                for action_name, action in actions_dict.items():
                    transition.prediction_probs[f'single_action_policy_{action_name}'] = np.zeros(len(actions_dict))
                    transition.prediction_probs[f'single_action_policy_{action_name}'][action] = 1.0

    # train the model
    if 'conservative_alpha' in model_params.keys():
        model_name = f"{model_params['num_layers']}__{model_params['num_hidden_neurons']}__{model_params['activation_func']}__{model_params['learning_rate']}__{model_params['n_steps']}__{model_params['conservative_alpha']}"
    else:
        model_name = f"{model_params['num_layers']}__{model_params['num_hidden_neurons']}__{model_params['activation_func']}__{model_params['learning_rate']}__{model_params['n_steps']}"
        
    model_path = os.path.join(models_folder, experiment_name, f'{model_name}.d3')
    os.makedirs(os.path.join(models_folder, experiment_name), exist_ok=True)

    rl_model = fit_dqn(train_dataset, algo, experiment_name, model_params)

    rl_model.save(model_path)
    rl_model.save_policy(model_path.replace('.d3', '_policy.pkl'))
    print('Model saved at:', model_path)

    # evaluate the model
    rl_policy = PolicyResolver(rl_model, list(actions_dict.values()))
    rl_greedy_policy = PolicyResolver(rl_model, list(actions_dict.values()), greedy=True)
    physician_policy = PolicyResolver('behavior_physician_policy', list(actions_dict.values()))
    physician_policy_greedy = PolicyResolver('behavior_physician_policy', list(actions_dict.values()), greedy=True)

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

    evaluation_results_1 = pd.concat([evaluation_results_1, pd.DataFrame(policy_eval_results, index=[0])] , ignore_index=True)

    # evaluate the behavior physician policy
    for eval_episodes, eval_name in zip([test_episodes, validation_episodes, train_episodes], ['test', 'validation', 'train']):
        wis, ci = weighted_importance_sampling_with_bootstrap(eval_episodes, 0.99, physician_policy, physician_policy, num_bootstrap_samples=1000, N=1000)
        evaluation_results_1[f'behavior_physician_policy_{eval_name}_wis'] = wis
        evaluation_results_1[f'behavior_physician_policy_{eval_name}_ci'] = str(ci)
        print(f"Weighted importance sampling for the behavior physician policy on the {eval_name} data: {wis}, CI: {ci}")

        # evaluate the greedy best physician policy
        wis, ci = weighted_importance_sampling_with_bootstrap(eval_episodes, 0.99, physician_policy_greedy, physician_policy, num_bootstrap_samples=1000, N=1000)
        evaluation_results_1[f'behavior_physician_policy_greedy_{eval_name}_wis'] = wis
        evaluation_results_1[f'behavior_physician_policy_greedy_{eval_name}_ci'] = str(ci)
        print(f"Weighted importance sampling for the greedy behavior physician policy on the {eval_name} data: {wis}, CI: {ci}")

        # evaluate the single action policies
        for action_name, action in actions_dict.items():
            single_action_policy = PolicyResolver(f'single_action_policy_{action_name}', list(actions_dict.values()))
            wis, ci = weighted_importance_sampling_with_bootstrap(eval_episodes, 0.99, single_action_policy, physician_policy, num_bootstrap_samples=1000, N=1000)
            evaluation_results_1[f'single_action_policy_{action_name}_{eval_name}_wis'] = wis
            evaluation_results_1[f'single_action_policy_{action_name}_{eval_name}_ci'] = str(ci)
            print(f"Weighted importance sampling for the single action policy {action_name} on the {eval_name} data: {wis}, CI: {ci}")

    return evaluation_results_1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CQL model for each group')
    parser.add_argument('--stratify_on', type=str, default='hospital', help='The stratification variable')
    args = parser.parse_args()
    stratify_on = args.stratify_on
    groups = trc.stratification_consts[stratify_on]['groups']

    main_data_folder = trc.stratification_consts[stratify_on]['processed_data']
    main_models_folder = trc.stratification_consts[stratify_on]['models']
    results_dir = os.path.join(trc.EXPERIMENTS_RESULTS, stratify_on)
    os.makedirs(results_dir, exist_ok=True)

    algo = 'CQL'
    reward_function = reward_func_mace_survival_repvasc

    num_layers_list = [1, 2]
    num_hidden_neurons_list = [32, 64]
    activation_func_list = ['leaky_relu', 'tanh']
    learning_rate_list = [0.01, 0.001]
    conservative_alpha_list = [0.01, 0.1, 0.5]
    n_steps_list = [300000]



    for group in groups:
    #    # TODO: remove this line
    #     if group == 'nstemi':
    #         continue
        data_folder = os.path.join(main_data_folder, group)
        models_folder = os.path.join(main_models_folder, group)
        experiment_name = f'{stratify_on}_{group}_{algo}_{reward_function.__name__}_{datetime.now().strftime("%Y%m%d%H%M%S")}'

        # dataframe to store the results
        evaluation_results = pd.DataFrame(columns=RESULTS_DF_COLS)

        # for num_layers, num_hidden_neurons, activation_func, learning_rate, n_steps, conservative_alpha in itertools.product(num_layers_list,
        #                                                                                                 num_hidden_neurons_list,
        #                                                                                                 activation_func_list,
        #                                                                                                 learning_rate_list,
        #                                                                                                 n_steps_list,
        #                                                                                                 conservative_alpha_list):
        def train_job_single(num_layers, num_hidden_neurons, activation_func, learning_rate, n_steps, conservative_alpha):    
            model_params = {
                'num_layers': num_layers,
                'num_hidden_neurons': num_hidden_neurons,
                'activation_func': activation_func,
                'learning_rate': learning_rate,
                'n_steps': n_steps,
                'conservative_alpha': conservative_alpha
            }

            evaluation_results_1 = run_training_session(data_folder, algo, model_params, models_folder, reward_function, stratify_on, experiment_name)

            return evaluation_results_1

            # # Concatenate the results and save
            # evaluation_results = pd.concat([evaluation_results, evaluation_results_1], ignore_index=True)
            # evaluation_results.to_csv(os.path.join(results_dir, f"{experiment_name}_evaluation_results_TEMP.csv"), index=False)

        # Generate all parameter combinations
        all_param_combinations = list(itertools.product(num_layers_list,
                                                        num_hidden_neurons_list,
                                                        activation_func_list,
                                                        learning_rate_list,
                                                        n_steps_list,
                                                        conservative_alpha_list))

        # Use joblib to parallelize the loop
        results = Parallel(n_jobs=5)(delayed(train_job_single)(*param_comb) for param_comb in all_param_combinations)

        # Concatenate all results into a single DataFrame
        evaluation_results = pd.concat(results, ignore_index=True)
        
        evaluation_results.to_csv(os.path.join(results_dir, f"{experiment_name}_evaluation_results.csv"), index=False)
        # os.remove(os.path.join(results_dir, f"{experiment_name}_evaluation_results_TEMP.csv"))
        print(f"Finished training a {algo} model for the {group} group")

    print("Finished training all models")


    