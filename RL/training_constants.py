import os

# kmeans clustering
n_clusters_list_checkup = range(10, 800, 1)
n_clusters_list_cath = range(2, 1000, 1)




# paths
major_save_dir = 'all_patients_data'
patient_list_csv = 'patiens_list.csv'
train_list_csv = 'obstructive_cad_train6.csv'
validation_list_csv = 'obstructive_cad_validation6.csv'
test_list_csv = 'obstructive_cad_test6.csv'
processed_data_path = 'processed_data_obstructive_cad6'
models_path = 'models_obstructive_cad6'
secondary_models_path = 'secondary_models_obstructive_cad6'
EXPERIMENTS_RESULTS = 'EXPERIMENTS_RESULTS2'
FIGURES_PATH = 'FIGURES2'
# models_path = 'models3'
rewards_list = ['survival', 'mace', 'repeated_revasc', 'cost_dollar']
treatments = ['CABG', 'Medical Therapy', 'PCI']

reference_cost = 18069  # cost of medical therapy ref: https://www.cmajopen.ca/content/4/3/E409
cost_ratios = {'CABG': 2.62, 'Medical Therapy': 1, 'PCI': 1.27}

death_followup_time = 365 * 3
outcome_followup_time = 365 * 3
min_acceptable_survival = 90  # minumum acceptable days of survival after the treatment (otherwise the patient is considered cardic death)
max_reward = 1e6

# behavior policy Kmeans model
n_clusters_behavior_policy = 177
experiment_type_behavior_policy = 'vanilla_kmeans'
behavior_policy_path = os.path.join(models_path,
                                    'vanilla_kmeans-2-999-all-caths-reward-mace-survival__20240502-214213_rl_policies',
                                    f'policies_{n_clusters_behavior_policy}.pkl')