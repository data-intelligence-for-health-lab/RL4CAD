import os

n_clusters_list_cath = range(20, 500, 1)
major_save_dir = 'RL/all_patients_data'
agg_strategy_path = 'RL/agg_strategy_v2.csv'

statified_patient_list_path = 'stratified_patient_lists'
processed_data_path = 'processed_data'
models_path = 'models'
EXPERIMENTS_RESULTS = 'experiments_results'

stratification_consts = {'facility': {
    'features_to_drop': ['Facility_FMC', 'Facility_RAH', 'Facility_UAH'],
    'name': 'Facility',
    'groups': ['calgary', 'edmonton'],
    'major_group': 'calgary',
    'patients_list': os.path.join(statified_patient_list_path, 'facility'),
    'processed_data': os.path.join(processed_data_path, 'facility'),
    'models': os.path.join(models_path, 'facility'),
},
    'hospital': {
        'features_to_drop': ['Facility_FMC', 'Facility_RAH', 'Facility_UAH'],
        'name': 'Hospital',
        'groups': ['FMC', 'RAH', 'UAH'],
        'major_group': 'FMC',
        'patients_list': os.path.join(statified_patient_list_path, 'hospital'),
        'processed_data': os.path.join(processed_data_path, 'hospital'),
        'models': os.path.join(models_path, 'hospital'),
    },
    'years': {
        'features_to_drop': [],
        'name': 'Years',
        'groups': ['2009-2016', '2017-2019'],
        'major_group': '2009-2016',
        'patients_list': os.path.join(statified_patient_list_path, 'years'),
        'processed_data': os.path.join(processed_data_path, 'years'),
        'models': os.path.join(models_path, 'years'),
    },
    
    'indication': {
        'features_to_drop': [
            "indicationtyp_Acute Coronary Syndrome",
            "indicationtyp_Congestive Heart Failure",
            "indicationtyp_Other",
            "indicationtyp_Stable Angina",
            "indicationtyp_Valvular Heart Disease",
            "indicationtyp_other",
            "indicationdx_NSTEMI",
            "indicationdx_Not Entered",
            "indicationdx_Unstable Angina"],
        'name': 'Indication',
        'groups': ['nstemi', 'stable-angina', 'unstable-angina'],
        'major_group': 'nstemi',
        'patients_list': os.path.join(statified_patient_list_path, 'indication'),
        'processed_data': os.path.join(processed_data_path, 'indication'),
        'models': os.path.join(models_path, 'indication'),
    },
    'sex': {
        'features_to_drop': ['Sex_Female'],
        'name': 'sex',
        'groups': ['male', 'female'],
        'major_group': ['male'],
        'patients_list': os.path.join(statified_patient_list_path, 'sex'),
        'processed_data': os.path.join(processed_data_path, 'sex'),
        'models': os.path.join(models_path, 'sex')
    }
}


# behavior policy Kmeans model
n_clusters_behavior_policy = 177
experiment_type_behavior_policy = 'vanilla_kmeans'

outcome_followup_time = 365 * 3
rewards_list = ['survival', 'mace', 'repeated_revasc', 'cost_dollar']
treatments = ['CABG', 'Medical Therapy', 'PCI']
min_acceptable_survival = 90  # minumum acceptable days of survival after the treatment (otherwise the patient is considered cardic death)


features_to_drop= ['neut',
                'triglycerides',
                'ldl cholesterol',
                'glucose random',
                'thyroid stimulating hormone',
                'urea',
                'hemoglobin a1c',
                'alkaline phosphatase',
                'magnesium',
                'calcium',
                'bilirubin total',
                'nrbc',
                'albumin',
                'ferritin']