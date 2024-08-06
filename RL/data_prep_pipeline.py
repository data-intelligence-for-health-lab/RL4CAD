import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import re
import miceforest as mf
from sklearn.cluster import KMeans
import training_constants as tc
import logging
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
from do_kmeans import do_kmeans

def aggregate_all_data(major_save_dir, agg_sterategy_df, rewards_list, sample_patients, processed_data_path):
    '''
    aggregate all data for all patients in the sample_patients.csv file
    :param major_save_dir: the directory that contains all patients' data
    :param agg_sterategy_df: the dataframe that contains the aggregation strategy
    :param sample_patients: the dataframe that contains the list of patients
    :param processed_data_path: the path to save the aggregated data
    :return: None
    '''

    # define the major CSV files
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    # checkup_csv_path = os.path.join(processed_data_path, 'all_checkups.csv')
    cath_csv_path = os.path.join(processed_data_path, 'all_caths.csv')
    death_csv_path = os.path.join(processed_data_path, 'all_deaths.csv')

    # cols_for_checkup = agg_sterategy_df[agg_sterategy_df['include_in_checkup'] == 1]['Variable_name'].to_list() + ['Procedure Number', 'SubsequentTreatment', 'Procedure Standard Time']
    cols_for_cath = agg_sterategy_df[agg_sterategy_df['include_in_cath'] == 1]['Variable_name'].to_list() + ['Procedure Number', 'SubsequentTreatment', 'Procedure Standard Time']
    cols_for_death = ['Procedure Number','Procedure Standard Time']

    csv_append = False
    for file_no in sample_patients['File No']:
        path = os.path.join(major_save_dir, str(file_no//1000), str(file_no), 'aggregated_df.csv')
        agg_df = pd.read_csv(path)

        # checkup_df = agg_df.loc[agg_df['Procedure Table'] == 'checkup', cols_for_checkup]
        cath_df = agg_df.loc[agg_df['Procedure Table'] == 'checkup_cath', cols_for_cath]
        death_df = agg_df.loc[agg_df['Procedure Table'] == 'death', cols_for_death]

        # checkup_df['File No'] = file_no
        cath_df['File No'] = file_no
        death_df['File No'] = file_no

        # drop the rows that 'Age_at_cath' is NaN
        # Reason: the 'Age_at_cath' is filled when we have any procedure for the patient, if not, all are NaN exccept icd and atc codes that are filled automatically
        # checkup_df = checkup_df.dropna(subset=['Age_at_cath'])
        cath_df = cath_df.dropna(subset=['Age_at_cath'])

        # drop the duplicates based on 'File No' and 'Procedure Standard Time'
        cath_df.drop_duplicates(subset=['File No', 'Procedure Standard Time'], keep='first')

        # add the reward columns
        for reward in rewards_list:
            reward_df = pd.read_csv(os.path.join(major_save_dir, str(file_no//1000), str(file_no), f'{reward}.csv'))
            if reward in ['survival', 'mace', 'repeated_revasc']:
                reward_df = reward_df.rename(columns={'Time to Event (days)': reward, 'checkup No': 'Procedure Number'})
                reward_df = reward_df[['Procedure Number', reward]]  # keep only the columns we need
                reward_df[reward] = reward_df[reward].fillna(tc.outcome_followup_time)  # fill the missing values with 3 years (maximum time we consider for survival)
                # reward_df[reward] = reward_df[reward].apply(lambda x: -100 if x < 90 else 100)  # if the patient is alive after 3 months, the reward is 100, otherwise -100
            elif reward == 'cost_dollar':
                reward_df = reward_df.rename(columns={'checkup No': 'Procedure Number'})
                reward_df['Time Earned (days)'] = reward_df['Time Earned (days)'].fillna(1)
                # reward_df['Time Earned (days)'].loc[reward_df['Time Earned (days)'] == 0] = 1  # if the time is 0, we consider it as 1 (to avoid division by zero)
                reward_df.loc[reward_df['Time Earned (days)'] == 0, 'Time Earned (days)'] = 1  # if the time is 0, we consider it as 1 (to avoid division by zero)
                reward_df['Dollar Cost'] = reward_df['Dollar Cost'].fillna(0)

                # reward_df[reward] = (reward_df['Dollar Cost'], reward_df['Time Earned (days)'])
                reward_df[reward] = [(x, y) for x, y in zip(reward_df['Dollar Cost'], reward_df['Time Earned (days)'])]
                reward_df = reward_df[['Procedure Number', reward]]
            else:
                raise ValueError(f"Invalid reward specified: {reward}")
            
            # checkup_df = checkup_df.merge(reward_df, on='Procedure Number', how='left')
            cath_df = cath_df.merge(reward_df, on='Procedure Number', how='left')

        if not csv_append:
            # checkup_df.to_csv(checkup_csv_path, index=False)
            cath_df.to_csv(cath_csv_path, index=False)
            death_df.to_csv(death_csv_path, index=False)
            csv_append = True
        else:
            # checkup_df.to_csv(checkup_csv_path, index=False, header=False, mode='a')
            cath_df.to_csv(cath_csv_path, index=False, header=False, mode='a')
            death_df.to_csv(death_csv_path, index=False, header=False, mode='a')


class DataPreprocessor:
    def __init__(self, agg_strategy_df):
        '''
        This class is used to preprocess the data and save the scaler for future use
        :param agg_strategy_df: the dataframe that contains the aggregation strategy

        '''
        self.agg_strategy_df = agg_strategy_df
        self.scalers = {}
        self.columns_to_drop = []

    def preprocess(self, df, data_type):
        '''
        preprocess the data in two modes of 'train' and 'test'.
        :param df: the dataframe that contains the data
        :param data_type: 'train' or 'test'
        :return: the preprocessed dataframe
        '''
        # Drop columns based on training data analysis
        if data_type == 'train':
            self.columns_to_drop = df.columns[df.isnull().mean() >= .3].tolist()
            df = df.drop(columns=self.columns_to_drop)
        elif data_type == 'test':
            df = df.drop(columns=self.columns_to_drop)
        else:
            raise ValueError("Invalid data type specified")

        # Normalize non-categorical columns
        non_categorical_cols = self.agg_strategy_df[self.agg_strategy_df['variable_type'].isin(['int', 'float', 'lab_float'])]['Variable_name'].values
        non_categorical_cols = [x for x in non_categorical_cols if x in df.columns]

        if data_type == 'train':
            scaler = MinMaxScaler()
            df[non_categorical_cols] = scaler.fit_transform(df[non_categorical_cols])
            self.scalers['scaler'] = scaler
        else:
            scaler = self.scalers['scaler']
            df[non_categorical_cols] = scaler.transform(df[non_categorical_cols])

        return df

    def save_preprocessor(self, path):
        '''
        save the scaler
        :param path: the path to save the preprocessor
        :return: None
        '''
        preprocessor_attributes = {'scaler': self.scalers['scaler'], 'columns_to_drop': self.columns_to_drop}
        with open(os.path.join(path, 'preprocessor.pkl'), 'wb') as f:
            pickle.dump(preprocessor_attributes, f)
    
    def load_preprocessor(self, path):
        '''
        load the scaler
        :param path: the path to load the preprocessor
        :return: None
        '''
        with open(os.path.join(path, 'preprocessor.pkl'), 'rb') as f:
            preprocessor_attributes = pickle.load(f)
            self.scalers['scaler'] = preprocessor_attributes['scaler']
            self.columns_to_drop = preprocessor_attributes['columns_to_drop']


def impute_and_save(df, data_name, models_path, random_state=100, load_existing=False):
    """
    Run MICE imputation on the given DataFrame and save the imputed data and kernel.
    If an existing kernel model is found and load_existing is True, it will be loaded instead.
    :param df: The DataFrame to impute
    :param data_name: The name of the data (cath or checkup)
    :param models_path: The path to save the imputed data and kernel
    :param random_state: The random state to use for MICE
    :param load_existing: If True, will load an existing kernel model if found (use for testing)
    :return: The imputed DataFrame
    """
    # sklearn's simple imputer
    imputer = SimpleImputer(strategy='median')
    if load_existing:
        imputer = pickle.load(open(os.path.join(models_path, f'imputer_{data_name}.pkl'), 'rb'))
    else:
        imputer.fit(df)
        with open(os.path.join(models_path, f'imputer_{data_name}.pkl'), 'wb') as f:
            pickle.dump(imputer, f)

    df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)
    return df_imputed



# def parallel_kmeans_job(n_cluster, train_data, test_data, val_data, dirs, kmeans_folder):
#     """
#     a function to help run K-means clustering on the train, test, and validation data in parallel.
#     """
#     # Unpack directories
#     train_dir, test_dir, validation_dir = dirs
    
#     # Run K-means clustering on the train data
#     run_kmeans_and_save([n_cluster], train_data, 'cath', train_dir, kmeans_folder, load_existing=False)
    
#     # Run K-means clustering on the test data
#     run_kmeans_and_save([n_cluster], test_data, 'cath', test_dir, kmeans_folder, load_existing=True)
    
#     # Run K-means clustering on the validation data
#     run_kmeans_and_save([n_cluster], val_data, 'cath', validation_dir, kmeans_folder, load_existing=True, verbose=True)

        
if __name__ == '__main__':       

    logging.basicConfig(filename='my_log_step2.log',  # Set the log file name
                        level=logging.DEBUG,     # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                        format='%(asctime)s [%(levelname)s]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info(f'Start the program at {datetime.datetime.now()}')

    major_save_dir = tc.major_save_dir
    agg_sterategy_df = pd.read_csv('agg_strategy_v2.csv')
    processed_data_path = tc.processed_data_path
    models_path = tc.models_path
    rewards_list = tc.rewards_list

    train_patients = pd.read_csv(tc.train_list_csv)
    validation_patients = pd.read_csv(tc.validation_list_csv)
    test_patients = pd.read_csv(tc.test_list_csv)
    train_dir = os.path.join(processed_data_path, 'train')
    validation_dir = os.path.join(processed_data_path, 'validation')
    test_dir = os.path.join(processed_data_path, 'test')

    # create the directories
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # aggregate all data
    aggregate_all_data(major_save_dir, agg_sterategy_df, rewards_list, train_patients, train_dir)
    aggregate_all_data(major_save_dir, agg_sterategy_df, rewards_list, validation_patients, validation_dir)
    aggregate_all_data(major_save_dir, agg_sterategy_df, rewards_list, test_patients, test_dir)

    logging.info(f'Aggregation complete at {datetime.datetime.now()}')

    # load the aggregated data
    cath_csv_train_path = os.path.join(train_dir, 'all_caths.csv')
    cath_df_train = pd.read_csv(cath_csv_train_path)
    cath_csv_val_path = os.path.join(validation_dir, 'all_caths.csv')
    cath_df_val = pd.read_csv(cath_csv_val_path)
    cath_csv_test_path = os.path.join(test_dir, 'all_caths.csv')
    cath_df_test = pd.read_csv(cath_csv_test_path)

    # remove unexpected duplicates based on File No and Procedure Number
    cath_df_train = cath_df_train.drop_duplicates(subset=['File No', 'Procedure Number'], keep='first')
    cath_df_val = cath_df_val.drop_duplicates(subset=['File No', 'Procedure Number'], keep='first')
    cath_df_test = cath_df_test.drop_duplicates(subset=['File No', 'Procedure Number'], keep='first')

    # shuffle the data and replace it with the original data
    cath_df_train = cath_df_train.sample(frac=1).reset_index(drop=True)
    cath_df_train.to_csv(cath_csv_train_path, index=False)
    cath_df_val = cath_df_val.sample(frac=1).reset_index(drop=True)
    cath_df_val.to_csv(cath_csv_val_path, index=False)
    cath_df_test = cath_df_test.sample(frac=1).reset_index(drop=True)
    cath_df_test.to_csv(cath_csv_test_path, index=False)


    # take out the 'File No' and 'Procedure Number' and 'SubsequentTreatment' columns
    cols_to_drop = ['File No', 'Procedure Number', 'SubsequentTreatment', 'Procedure Standard Time'] + rewards_list
    cath_df_train = cath_df_train.drop(columns=cols_to_drop)
    cath_df_val = cath_df_val.drop(columns=cols_to_drop)
    cath_df_test = cath_df_test.drop(columns=cols_to_drop)



    # preprocess and scaling
    preprocessor = DataPreprocessor(agg_sterategy_df)
    cath_df_train_processed = preprocessor.preprocess(cath_df_train, 'train')
    preprocessor.save_preprocessor(models_path)

    # Load scaler and preprocess test data
    preprocessor.load_preprocessor(models_path)
    cath_df_val_processed = preprocessor.preprocess(cath_df_val, 'test')
    cath_df_test_processed = preprocessor.preprocess(cath_df_test, 'test')

    logging.info(f'Preprocessing complete at {datetime.datetime.now()}')

    # run MICE imputation and save the imputed data and kernel
    load_existing = False
    cath_df_train_imputed = impute_and_save(cath_df_train_processed, 'cath', models_path, load_existing=load_existing)
    cath_df_train_imputed.to_csv(os.path.join(train_dir, 'all_caths_train_imputed.csv'), index=False)

    logging.info(f'MICE imputation (train) complete at {datetime.datetime.now()}')

    # run MICE on the test data
    cath_df_test_imputed = impute_and_save(cath_df_test_processed, 'cath', models_path, load_existing=True)
    cath_df_test_imputed.to_csv(os.path.join(test_dir, 'all_caths_test_imputed.csv'), index=False)
    cath_df_val_imputed = impute_and_save(cath_df_val_processed, 'cath', models_path, load_existing=True)
    cath_df_val_imputed.to_csv(os.path.join(validation_dir, 'all_caths_validation_imputed.csv'), index=False)

    logging.info(f'MICE imputation (test) complete at {datetime.datetime.now()}')

    # run K-means clustering on the imputed data
    do_kmeans()
    # folder to save the kmeans models
    # kmeans_folder = os.path.join(models_path, 'vanilla_kmeans_models')
    # if not os.path.exists(kmeans_folder):
    #     os.makedirs(kmeans_folder)

    # kmeans clustering
    # n_clusters_list_checkup = tc.n_clusters_list_checkup
    # n_clusters_list_cath = tc.n_clusters_list_cath
    # kmeans_dirs = (train_dir, test_dir, validation_dir)

    # task_func = functools.partial(parallel_kmeans_job, train_data=cath_df_train_imputed, test_data=cath_df_test_imputed, val_data=cath_df_val_imputed, dirs=kmeans_dirs, kmeans_folder=kmeans_folder)

    # with ProcessPoolExecutor(max_workers=50) as executor:
    #     futures = [executor.submit(task_func, n_cluster) for n_cluster in n_clusters_list_cath]
    #     for future in as_completed(futures):
    #         try:
    #             future.result()
    #         except Exception as e:
    #             print(f'Error during kmeans clustering: {e}')

    # for n_cluster in n_clusters_list_cath:
    #     # run K-means clustering on the train data
    #     run_kmeans_and_save([n_cluster], cath_df_train_imputed, 'cath', train_dir, kmeans_folder, load_existing=False)

    #     # run K-means clustering on the test data
    #     run_kmeans_and_save([n_cluster], cath_df_test_imputed, 'cath', test_dir, kmeans_folder, load_existing=True)
    #     run_kmeans_and_save([n_cluster], cath_df_val_imputed, 'cath', validation_dir, kmeans_folder, load_existing=True)

    logging.info(f'K-means clustering (train and test) complete at {datetime.datetime.now()}')


    print('Aggregation, preprocessing, imputation, and clustering complete.')
