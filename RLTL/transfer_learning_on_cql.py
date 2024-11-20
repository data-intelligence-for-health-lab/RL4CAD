"""
This script evaluates the CQL policy of one group on a different group's data (e.g., CQL policy of Calgary physicians on Edmonton data or vice versa).
"""
import d3rlpy
import pandas as pd
import os
import numpy as np
import random
import pickle
from datetime import datetime
from sklearn import preprocessing
from joblib import Parallel, delayed
import argparse
import transfer_constants as trc
from rl_utils import get_episodes, get_d3rlpy_dataset, PolicyResolver, weighted_importance_sampling_with_bootstrap, reward_func_mace_survival_repvasc, ClusteringBasedInference

def finetune_cql(base_model, episodes, model_params, n_steps, actions_dict, experiment_name):
    """
    Finetune a CQL model on a new dataset
    """
    finetune_dataset = get_d3rlpy_dataset(episodes, actions_dict)

    num_layers = model_params['num_layers']
    num_hidden_neurons = model_params['num_hidden_neurons']
    activation_func = model_params['activation_func']
    learning_rate = model_params['learning_rate']
    learning_rate = learning_rate / 10  # reduce the learning rate for finetuning
    conservative_alpha = model_params['conservative_alpha']
    # conservative_alpha = conservative_alpha / 2  # reduce the conservative alpha for finetuning

    # encoder factory
    encoder_factory = d3rlpy.models.VectorEncoderFactory(
        hidden_units=[num_hidden_neurons] * num_layers,
        activation=activation_func)
    
    new_cql = d3rlpy.algos.DiscreteCQLConfig(alpha=conservative_alpha, encoder_factory=encoder_factory, learning_rate=learning_rate).create(device='cuda')
    # new_cql = d3rlpy.algos.DQNConfig(encoder_factory=encoder_factory, learning_rate=learning_rate).create(device='cuda')
    new_cql.build_with_dataset(finetune_dataset)
    new_cql.copy_q_function_from(base_model)
    # train the RL model
    new_cql.fit(
        finetune_dataset,
        n_steps=n_steps,
        experiment_name= experiment_name,show_progress=False, save_interval=1e9)
    
    return new_cql
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the CQL policy of one group on the other group data')
    parser.add_argument('--stratify_on', type=str, default='hospital', help='Stratify on which feature?')
    parser.add_argument('--do_for_all', action='store_true', help='Evaluate the policy of all groups on the data of the major group')

    args = parser.parse_args()
    stratify_on = args.stratify_on
    do_for_all = args.do_for_all

    groups = trc.stratification_consts[stratify_on]['groups']
    major_group = trc.stratification_consts[stratify_on]['major_group']

    rewards_list = trc.rewards_list
    reward_function = reward_func_mace_survival_repvasc

    main_data_folder = trc.stratification_consts[stratify_on]['processed_data']
    main_models_folder = trc.stratification_consts[stratify_on]['models']
    behavior_experiment_type = trc.experiment_type_behavior_policy

    # encode the actions
    treatments = trc.treatments
    action_encoder = preprocessing.LabelEncoder()
    action_encoder.fit(np.array(treatments).reshape(-1, 1))
    actions_dict = dict(zip(action_encoder.classes_, action_encoder.transform(action_encoder.classes_)))
    print("Actions dictionary:", actions_dict)

    # find selected CQL models for each group
    selected_hyp_df = pd.read_csv(os.path.join(trc.EXPERIMENTS_RESULTS, stratify_on, 'selected_cql_hyperparameters.csv'))

    # # manually set the CQL model paths for each group
    # cql_models_dict = {'calgary': 'models/facility/calgary/facility_calgary_CQL_reward_func_mace_survival_repvasc_20240721005228/2__64__relu__0.01__500000__0.01.d3',
    #                    'edmonton': 'models/facility/edmonton/facility_edmonton_CQL_reward_func_mace_survival_repvasc_20240721013000/2__64__relu__0.01__500000__0.01.d3'
    # }
    # model_params = {
    #     'num_layers': 2,
    #     'num_hidden_neurons': 64,
    #     'activation_func': 'relu',
    #     'learning_rate': 0.001,
    #     'conservative_alpha': 0.01
    # }
    n_steps = 100000  # 10 epochs
    # n_steps = 10000  # 1 epoch
    n_repeats = 5
    portions_list = [0, 0.01, 0.1, 0.2, 0.5, 1.0]

    behavior_policy_dict = {}
    train_episodes_dict = {}
    test_episodes_dict = {}
    for group in groups:

        # load the behavior policy
        data_folder_grp = os.path.join(main_data_folder, group)
        models_folder_grp = os.path.join(main_models_folder, group)
        behavior_policy_path = os.path.join(models_folder_grp, f"behavior_policy_{behavior_experiment_type}.pkl")
        behavior_policy_data = pickle.load(open(behavior_policy_path, 'rb'))
        behavior_n_clusters = behavior_policy_data['n_clusters']
        behavior_policy_cluster_model_path = os.path.join(models_folder_grp, f"{behavior_experiment_type}_models", f"{behavior_experiment_type}_{behavior_n_clusters}.pkl")
        behavior_policy_cluster_model = pickle.load(open(behavior_policy_cluster_model_path, 'rb'))
        behavior_policy_dict[group] = ClusteringBasedInference(behavior_policy_cluster_model,
                                                               behavior_policy_data['n_clusters'],
                                                               behavior_policy_data['behavior_policy'])
        # load the training data
        all_caths_train = pd.read_csv(os.path.join(data_folder_grp, 'train', 'all_caths.csv'))
        all_caths_train['SubsequentTreatment'].fillna('Medical Therapy', inplace=True)
        all_caths_train_imputed = pd.read_csv(os.path.join(data_folder_grp, 'train', 'all_caths_train_imputed.csv'))

        # load the test data
        all_caths_test = pd.read_csv(os.path.join(data_folder_grp, 'test', 'all_caths.csv'))
        all_caths_test['SubsequentTreatment'].fillna('Medical Therapy', inplace=True)
        all_caths_test_imputed = pd.read_csv(os.path.join(data_folder_grp, 'test', 'all_caths_test_imputed.csv'))

        # drop feature containing the groups' information
        group_features_to_drop = trc.stratification_consts[stratify_on]['features_to_drop']
        all_caths_test.drop(columns=group_features_to_drop, inplace=True)
        all_caths_test_imputed.drop(columns=group_features_to_drop, inplace=True)
        all_caths_train.drop(columns=group_features_to_drop, inplace=True)
        all_caths_train_imputed.drop(columns=group_features_to_drop, inplace=True)

        # get the test episodes
        train_episodes = get_episodes(all_caths_train, all_caths_train_imputed, action_encoder, rewards_list, reward_function)
        test_episodes = get_episodes(all_caths_test, all_caths_test_imputed, action_encoder, rewards_list, reward_function)

        # put behavior policy in the test episodes
        for episodes_dataset in [train_episodes, test_episodes]:
            for episode in episodes_dataset:
                for transition in episode:
                    transition.prediction_probs['behavior_policy'] = behavior_policy_dict[group].get_policy(transition.state.reshape(1, -1))[0,:].tolist()

        test_episodes_dict[group] = test_episodes
        train_episodes_dict[group] = train_episodes

    # evaluate the behavior policy of one group on the other group's data
    results_df = pd.DataFrame(columns=['evaluation_on', 'alpha', 'data_portion', 'wis', 'ci', 'greedy_wis', 'greedy_ci'])
    result_filename = os.path.join(trc.EXPERIMENTS_RESULTS, stratify_on, f"{stratify_on}_evaluate_on_finetuned_vlowerLRloweralpha_{major_group}_CQL_policy_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
    
    which_groups_of_policy = groups if do_for_all else [major_group]
    for group_of_policy in which_groups_of_policy:  # only evaluate the major group's policy
        for alpha in selected_hyp_df['alpha'].unique():
            selected_cql_model = selected_hyp_df[(selected_hyp_df['group'] == group_of_policy) & (selected_hyp_df['alpha'] == alpha)].iloc[0]
            model_path = os.path.join(trc.stratification_consts[stratify_on]['models'], group_of_policy, selected_cql_model['experiment_name'])
            model_path = os.path.join(model_path, f"{selected_cql_model['num_layers']}__{selected_cql_model['num_hidden_neurons']}__{selected_cql_model['activation_func']}__{selected_cql_model['learning_rate']}__{selected_cql_model['n_steps']}__{selected_cql_model['alpha']}.d3")
            model_params = {
                'num_layers': int(selected_cql_model['num_layers']),
                'num_hidden_neurons': int(selected_cql_model['num_hidden_neurons']),
                'activation_func': selected_cql_model['activation_func'],
                'learning_rate': selected_cql_model['learning_rate'],
                'conservative_alpha': selected_cql_model['alpha']
            }
            opposing_rl_model = d3rlpy.load_learnable(model_path)
            for group_of_data in groups:
                if group_of_policy == group_of_data:
                    continue
                print(f"Evaluating the finetuned CQL policy of {group_of_policy} on the data of {group_of_data}")
                test_episodes = test_episodes_dict[group_of_data]
                train_episodes = train_episodes_dict[group_of_data]

                policy_eval_results = {}
                policy_eval_results['evaluation_on'] = group_of_data
                policy_eval_results['alpha'] = alpha
                behavior_pi_B = PolicyResolver('behavior_policy',  list(actions_dict.values()))


                # finetune the CQL model of the group_of_policy on the data of group_of_data
                # for portion in [0.01, 0.1, 0.2, 0.5, 1.0]:
                def finetune_and_evaluate_on_portion(portion):
                    n_repeats_for_portion = 1 if (portion == 1.0 or portion == 0) else n_repeats
                    for repeat in range(n_repeats_for_portion):
                        print(f"Finetuning the CQL model of {group_of_policy} on the data of {group_of_data} with {portion} of the data - repeat {repeat}")
                        policy_eval_results['data_portion'] = portion
                        
                        if portion != 0:
                            # choose a random portion of the training data
                            random.sample(train_episodes, int(portion * len(train_episodes)))
                            finetuned_model = finetune_cql(opposing_rl_model, train_episodes, model_params, n_steps, actions_dict, f"{group_of_policy}_finetuned_on_{group_of_data}_portion_{portion}")
                        else:
                            finetuned_model = opposing_rl_model  # no finetuning

                        # evaluate the finetuned model
                        finetuned_pi_RL = PolicyResolver(finetuned_model,  list(actions_dict.values()))
                        greedy_finetuned_pi_RL = PolicyResolver(finetuned_model,  list(actions_dict.values()), greedy=True)

                        wis, ci = weighted_importance_sampling_with_bootstrap(test_episodes, 0.99, finetuned_pi_RL, behavior_pi_B, num_bootstrap_samples=1000, N=1000)
                        if repeat == 0:
                            policy_eval_results[f"wis"] = wis
                            policy_eval_results[f"ci"] = list(ci)
                        else:
                            policy_eval_results[f"wis"] += wis
                            policy_eval_results[f"ci"][0] += ci[0]
                            policy_eval_results[f"ci"][1] += ci[1]

                        wis, ci = weighted_importance_sampling_with_bootstrap(test_episodes, 0.99, greedy_finetuned_pi_RL, behavior_pi_B, num_bootstrap_samples=1000, N=1000)
                        if repeat == 0:
                            policy_eval_results[f"greedy_wis"] = wis
                            policy_eval_results[f"greedy_ci"] = list(ci)
                        else:
                            policy_eval_results[f"greedy_wis"] += wis
                            policy_eval_results[f"greedy_ci"][0] += ci[0]
                            policy_eval_results[f"greedy_ci"][1] += ci[1]

                    policy_eval_results[f"wis"] /= n_repeats_for_portion
                    policy_eval_results[f"ci"][0] /= n_repeats_for_portion
                    policy_eval_results[f"ci"][1] /= n_repeats_for_portion
                    policy_eval_results[f"ci"] = tuple(policy_eval_results[f"ci"])

                    policy_eval_results[f"greedy_wis"] /= n_repeats_for_portion
                    policy_eval_results[f"greedy_ci"][0] /= n_repeats_for_portion
                    policy_eval_results[f"greedy_ci"][1] /= n_repeats_for_portion
                    policy_eval_results[f"greedy_ci"] = tuple(policy_eval_results[f"greedy_ci"])

                    return policy_eval_results
                
                # parallelize the evaluation on different portions of the data
                results = Parallel(n_jobs=6)(delayed(finetune_and_evaluate_on_portion)(portion) for portion in portions_list)
                for policy_eval_results in results:
                    # save the results
                    print(policy_eval_results)
                    results_df = pd.concat([results_df, pd.DataFrame([policy_eval_results])])
                results_df.to_csv(os.path.join(trc.EXPERIMENTS_RESULTS, stratify_on, f"{stratify_on}_evaluate_on_finetuned_{major_group}_CQL_policy_temp.csv"), index=False)

        results_df.to_csv(result_filename, index=False)
        os.remove(os.path.join(trc.EXPERIMENTS_RESULTS, stratify_on, f"{stratify_on}_evaluate_on_finetuned_{major_group}_CQL_policy_temp.csv"))
        print("Done!")

