import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import re
from sklearn.cluster import KMeans
import transfer_constants as trc
import logging
import datetime
import argparse
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import functools
# from joblib import Parallel, delayed



def run_kmeans_and_save(n_clusters_list, data_df, dataset_name, processed_data_path, kmeans_folder, random_state=100, load_existing=False, verbose=False):
    '''
    run kmeans clustering on the data and save the clusters
    :param n_clusters_list: the list of number of clusters
    :param data_df: the dataframe that contains the data
    :param dataset_name: the name of the dataset (cath or checkup)
    :param processed_data_path: the path to save the clusters index
    :param kmeans_folder: the folder to save the kmeans models
    :return: None
    '''
    for n_clusters in n_clusters_list:
        kmeans_filename = f'kmeans_{dataset_name}_{n_clusters}.pkl'
        kmeans_filepath = os.path.join(kmeans_folder, kmeans_filename)
        if load_existing:
            with open(kmeans_filepath, 'rb') as f:
                kmeans = pickle.load(f)
        else:
            max_iter = max(500, 10 * n_clusters)
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, init='k-means++', n_init='auto', max_iter=max_iter).fit(data_df)
            with open(kmeans_filepath, 'wb') as f:
                pickle.dump(kmeans, f)

        clusters = kmeans.predict(data_df)
        clusters_df = pd.DataFrame(clusters, columns=['cluster'])
        clusters_df.to_csv(os.path.join(processed_data_path, f'{dataset_name}_kmeans_clusters_{n_clusters}.csv'), index=False)

        if verbose:
            print(f'K-means clustering for {dataset_name} with {n_clusters} clusters complete and file saved.')


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
                reward_df[reward] = reward_df[reward].fillna(trc.outcome_followup_time)  # fill the missing values with 3 years (maximum time we consider for survival)
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
            # self.columns_to_drop = df.columns[df.isnull().mean() >= .3].tolist()
            self.columns_to_drop = trc.features_to_drop  # drop the features with high missing rates (based on the previous research)
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

def prepare_data(stratify_on=None):
    """
    prepare the data for the RL model

    :param stratification_on: the stratification variable
    :return: None
    """
    # set the logger
    logging.basicConfig(filename='prepare_data.log', level=logging.INFO)
    logging.info(f"Start preparing the data at {datetime.datetime.now()}")
    
    # read the aggregation strategy
    agg_strategy_df = pd.read_csv(trc.agg_strategy_path) 

    # read the rewards list
    rewards_list = trc.rewards_list 

    # read the sample patients
    patient_list_folder = trc.stratification_consts[stratify_on]['patients_list']
    groups = trc.stratification_consts[stratify_on]['groups']

    # make the directories
    major_save_dir = trc.major_save_dir
    processed_data_path = os.path.join(trc.processed_data_path, stratify_on)
    os.makedirs(processed_data_path, exist_ok=True)
    models_path = os.path.join(trc.models_path, stratify_on)

    for group in groups:
        group_processed_data_dir = os.path.join(processed_data_path, group)
        os.makedirs(group_processed_data_dir, exist_ok=True)
        group_models_path = os.path.join(models_path, group)
        os.makedirs(group_models_path, exist_ok=True)

        for dataset in ['train', 'test', 'validation']:
            dataset_dir = os.path.join(group_processed_data_dir, dataset)
            os.makedirs(dataset_dir, exist_ok=True)

            # read the sample patients
            sample_patients = pd.read_csv(os.path.join(patient_list_folder, f'{group}_{dataset}.csv'))
            aggregate_all_data(major_save_dir, agg_strategy_df, rewards_list, sample_patients, dataset_dir)
            logging.info(f"Data for {group} {dataset} is aggregated. stratification_on: {stratify_on} - {datetime.datetime.now()}")

            # load the aggregated data
            cath_df = pd.read_csv(os.path.join(dataset_dir, 'all_caths.csv'))

            # remove unexpected duplicates based on File No and Procedure Number
            cath_df.drop_duplicates(subset=['File No', 'Procedure Number'], keep='first', inplace=True)

            # shuffle the data and replace it with the original data
            cath_df = cath_df.sample(frac=1, random_state=100).reset_index(drop=True)

            # save the shuffled data
            cath_df.to_csv(os.path.join(dataset_dir, 'all_caths.csv'), index=False)

            # take out the non-feature columns
            cols_to_drop = ['File No', 'Procedure Number', 'SubsequentTreatment', 'Procedure Standard Time'] + rewards_list
            cath_df = cath_df.drop(columns=cols_to_drop)

            # preprocess and scaling
            if dataset == 'train':
                preprocessor = DataPreprocessor(agg_strategy_df)
                cath_df = preprocessor.preprocess(cath_df, data_type='train')
                preprocessor.save_preprocessor(group_models_path)
            else:
                preprocessor = DataPreprocessor(agg_strategy_df)
                preprocessor.load_preprocessor(group_models_path)
                cath_df = preprocessor.preprocess(cath_df, data_type='test')

            logging.info(f"Data for {group} {dataset} is preprocessed. stratification_on: {stratify_on} - {datetime.datetime.now()}")

            # impute the missing values
            if dataset == 'train':
                cath_df_imputed = impute_and_save(cath_df, 'cath', group_models_path, load_existing=False)
            else:
                cath_df_imputed = impute_and_save(cath_df, 'cath', group_models_path, load_existing=True)

            # save the imputed data
            cath_df_imputed.to_csv(os.path.join(dataset_dir, f'all_caths_{dataset}_imputed.csv'), index=False)
            logging.info(f"Data for {group} {dataset} is prepared. stratification_on: {stratify_on} - {datetime.datetime.now()}")

            # # Kmeans clustering
            # experiment_type = trc.experiment_type_behavior_policy
            # n_clusters = trc.n_clusters_behavior_policy
            # os.makedirs(os.path.join(dataset_dir, experiment_type), exist_ok=True)
            # os.makedirs(os.path.join(group_models_path, f'{experiment_type}_models'), exist_ok=True)
            # if dataset == 'train':
            #     run_kmeans_and_save([n_clusters], cath_df_imputed, experiment_type,
            #                         os.path.join(dataset_dir, experiment_type),
            #                         os.path.join(group_models_path, f'{experiment_type}_models'),
            #                         load_existing=False, verbose=True)
            # else:
            #     run_kmeans_and_save([n_clusters], cath_df_imputed, experiment_type,
            #                         os.path.join(dataset_dir, experiment_type),
            #                         os.path.join(group_models_path, f'{experiment_type}_models'),
            #                         load_existing=True, verbose=True)
                
            # logging.info(f"Kmeans clustering for {group} {dataset} is done. stratification_on: {stratify_on} - {datetime.datetime.now()}")

    logging.info(f"Data preparation is done. stratification_on: {stratify_on} - {datetime.datetime.now()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare the data for the RL/TL study')
    parser.add_argument('--stratify_on', type=str, default='hospital', help='The stratification variable')
    args = parser.parse_args()
    stratification_on = args.stratify_on
    prepare_data(stratification_on)

    print(f'Data preparation is done for {stratification_on} stratification.')