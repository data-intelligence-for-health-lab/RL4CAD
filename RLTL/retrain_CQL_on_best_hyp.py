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
from train_cql import run_training_session

RESULTS_DF_COLS = ['experiment_name', 'num_layers', 'num_hidden_neurons', 'activation_func', 'learning_rate', 'n_steps',
                                               'rl_policy_test_wis', 'rl_policy_test_ci',
                                               'rl_policy_validation_wis', 'rl_policy_validation_ci',
                                               'rl_policy_train_wis', 'rl_policy_train_ci',
                                               'rl_greedy_policy_test_wis', 'rl_greedy_policy_test_ci',
                                               'rl_greedy_policy_validation_wis', 'rl_greedy_policy_validation_ci',
                                                'rl_greedy_policy_train_wis', 'rl_greedy_policy_train_ci']


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

    # find selected CQL models for each group
    selected_hyp_df = pd.read_csv(os.path.join(trc.EXPERIMENTS_RESULTS, stratify_on, 'selected_cql_hyperparameters.csv'))
    n_steps = 1000000  # number of training steps with the best hyperparameters



    for group in groups:

        data_folder = os.path.join(main_data_folder, group)
        models_folder = os.path.join(main_models_folder, group)
        experiment_name = f'{stratify_on}_{group}_{algo}_{reward_function.__name__}_BEST_HYPERPARAMS'

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
                'num_layers': int(num_layers),
                'num_hidden_neurons': int(num_hidden_neurons),
                'activation_func': activation_func,
                'learning_rate': learning_rate,
                'n_steps': int(n_steps),
                'conservative_alpha': conservative_alpha
            }

            evaluation_results_1 = run_training_session(data_folder, algo, model_params, models_folder, reward_function, stratify_on, experiment_name)

            return evaluation_results_1

            # # Concatenate the results and save
            # evaluation_results = pd.concat([evaluation_results, evaluation_results_1], ignore_index=True)
            # evaluation_results.to_csv(os.path.join(results_dir, f"{experiment_name}_evaluation_results_TEMP.csv"), index=False)

        # Generate all parameter combinations
        all_param_combinations = []
        for alpha in selected_hyp_df['alpha'].unique():
            selected_cql_model = selected_hyp_df[(selected_hyp_df['group'] == group) & (selected_hyp_df['alpha'] == alpha)].iloc[0]
            all_param_combinations.append((selected_cql_model['num_layers'], selected_cql_model['num_hidden_neurons'], selected_cql_model['activation_func'], selected_cql_model['learning_rate'], n_steps, selected_cql_model['alpha']))

        # Use joblib to parallelize the loop
        results = Parallel(n_jobs=5)(delayed(train_job_single)(*param_comb) for param_comb in all_param_combinations)

        # Concatenate all results into a single DataFrame
        evaluation_results = pd.concat(results, ignore_index=True)
        
        evaluation_results.to_csv(os.path.join(results_dir, f"{experiment_name}_evaluation_results.csv"), index=False)
        # os.remove(os.path.join(results_dir, f"{experiment_name}_evaluation_results_TEMP.csv"))
        print(f"Finished training a {algo} model for the {group} group")

    print("Finished training all models")