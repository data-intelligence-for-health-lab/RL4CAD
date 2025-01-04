import os

import numpy as np
import json
import time
import pickle
import pandas as pd
from types import SimpleNamespace
import psycopg2
import APPROACH.constants as cons
from os.path import join
from APPROACH.summarize_HRQOL import summarize_EQ5D, summarize_SAQ
from APPROACH.lab_data_cleaning_helper import LabDataCleaningHelper
import APPROACH.approach_cleaning_helper as approach_helper
import APPROACH.ICDcodeHelper as icd
from APPROACH.PandasCadExtension import JsonForCadAccessor

pd.options.mode.chained_assignment = None  # default='warn'


class ApproachSqlDataset:
    def __init__(self, schema_name, table_name, patient_id_column):
        """
        This class makes a object that contain everything required to connect to the APPROACH SQL database
        :param schema_name:
        :param table_name:
        """
        with open(cons.SQL_CREDENTIALS, 'r') as f:
            credentials = json.load(f)

        # Connect to the PostgreSQL database
        self.conn = psycopg2.connect(
            host=credentials['host'],
            port=credentials['port'],
            database="CAD",
            user=credentials['username'],
            password=credentials['password']
        )
        self.schema_name = schema_name
        self.table_name = table_name
        self.patient_id_column = patient_id_column
        self.cursor = self.conn.cursor()

    def __del__(self):
        self.cursor.close()
        self.conn.close()

    def execute_query(self, query):
        """
        This function executes a query and returns the result as a pandas DataFrame
        :param query:
        :return:
        """
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        column_names = [desc[0] for desc in self.cursor.description]
        df = pd.DataFrame(rows, columns=column_names)
        return df


"""
This class is a class of the whole APPROACH cohort. It opens the data files and does the main processes
"""


class Cohort:

    def __init__(self, df_names_to_open='all'):
        self.data = self.Data()
        self.open_csv_files(df_names=df_names_to_open)

        # handle approach CATH data
        self.all_patients = self.data.cath.df['File No'].unique()
        self.data.cath.df = approach_helper.clean_cath(self.data.cath.df)
        self.data.carat_main.df = approach_helper.clean_carat_main(self.data.carat_main.df)
        self.data.cath.df = approach_helper.connect_cath_and_carat(self.data.cath.df, self.data.carat_main.df)

        self.all_procedures = pd.DataFrame()

    # this class is a list of Databases in the cohort
    class Data:
        def __init__(self):
            """
            This class is a list of Databases in the cohort
            each database is a SimpleNamespace object with the following attributes:
            *param* df: a pandas DataFrame. it can be empty if the database is not opened.
            *param* sql_handler: a SqlHandler object. it can be None if we load the data from csv files.
            *param* csv_path: the path to the csv file. it can be None if we load the data from SQL database.
            *param* name: the name of the database. it is used to identify the database.
            *param* encoding: the encoding of the csv file. it is used when we load the data from csv files.
            *param* standard_time: a tuple of two elements. the first element is the name of the column that contains
             the time and the second element is the format of the time. it is used to standardize the time format.

            """
            # APPROACH CAD databases
            self.cath = SimpleNamespace(df=pd.DataFrame([]), sql_handler=None,
                                        csv_path=join(cons.DATASET_FOLDER, cons.CATH_DATA_FILE),
                                        name='cath',
                                        encoding='utf-8',
                                        standard_time=('CATH Date', None))
            self.pci = SimpleNamespace(df=pd.DataFrame([]), sql_handler=None,
                                       csv_path=join(cons.DATASET_FOLDER, cons.PCI_DATA_FILE),
                                       name='pci',
                                       encoding='utf-8',
                                       standard_time=('PCI Date', None))
            self.cabg = SimpleNamespace(df=pd.DataFrame([]), sql_handler=None,
                                        csv_path=join(cons.DATASET_FOLDER, cons.CABG_DATA_FILE),
                                        name='cabg',
                                        encoding='utf-8',
                                        standard_time=('Procedure Start Date', None))
            self.carat_main = SimpleNamespace(df=pd.DataFrame([]), sql_handler=None,
                                              csv_path=join(cons.DATASET_FOLDER, cons.CARAT_MAIN_DATA_FILE),
                                              name='carat_main',
                                              encoding='utf-8',
                                              standard_time=('ProcedureDate', None))

            # vital statistics from ADMIN
            self.vs = SimpleNamespace(df=pd.DataFrame([]), sql_handler=None,
                                      csv_path=join(cons.ADMIN_DATASET_FOLDER, cons.ADMIN_VS_FILE),
                                      name='vs',
                                      encoding='iso-8859-1',
                                      standard_time=('DETHDATE', '%d%b%Y:%H:%M:%S'))
            # discharge abstract database from ADMIN
            self.dad = SimpleNamespace(df=pd.DataFrame([]), sql_handler=None,
                                       csv_path=join(cons.ADMIN_DATASET_FOLDER, cons.ADMIN_DAD_FILE),
                                       name='dad',
                                       encoding='iso-8859-1',
                                       standard_time=('ADMIT_DTTM', '%d%b%Y:%H:%M:%S'))
            # National Ambulatory Care Reporting System from ADMIN
            self.nacrs = SimpleNamespace(df=pd.DataFrame([]), sql_handler=None,
                                         csv_path=join(cons.ADMIN_DATASET_FOLDER, cons.ADMIN_NACRS_FILE),
                                         name='nacrs',
                                         encoding='iso-8859-1',
                                         standard_time=('VISIT_DTTM', '%d%b%Y:%H:%M:%S'))
            # ACCS from ADMIN
            self.accs = SimpleNamespace(df=pd.DataFrame([]), sql_handler=None,
                                        csv_path=join(cons.ADMIN_DATASET_FOLDER, cons.ADMIN_ACCS_FILE),
                                        name='accs',
                                        encoding='iso-8859-1',
                                        standard_time=('VISDATE', '%Y%m%d'))
            # CLAIMS from ADMIN -- get from SQL DB
            sql_handler_claims = ApproachSqlDataset(schema_name='public',
                                                    table_name='CLAIMS',
                                                    patient_id_column='FILE_NO')
            self.claims = SimpleNamespace(df=pd.DataFrame([]), sql_handler=sql_handler_claims,
                                          csv_path=None,
                                          name='claims',
                                          encoding='iso-8859-1',
                                          standard_time=('SE_START_DATE', '%d%b%Y:%H:%M:%S'))
            # LAB from ADMIN -- get from SQL DB
            sql_handler_lab = ApproachSqlDataset(schema_name='public',
                                                 table_name='LAB',
                                                 patient_id_column='FILE_NO')
            self.lab = SimpleNamespace(df=pd.DataFrame([]), sql_handler=sql_handler_lab,
                                       csv_path=None,
                                       name='lab',
                                       encoding='iso-8859-1',
                                       standard_time=('TEST_VRFY_DTTM', '%d%b%Y:%H:%M:%S'))
            # Pharmaceutical Information Network(PIN) from ADMIN -- get from SQL DB
            sql_handler_pin = ApproachSqlDataset(schema_name='public',
                                                 table_name='PIN',
                                                 patient_id_column='FILE_NO')
            self.pin = SimpleNamespace(df=pd.DataFrame([]), sql_handler=sql_handler_pin,
                                       csv_path=None,
                                       name='pin',
                                       encoding='iso-8859-1',
                                       standard_time=('DSPN_DATE', '%d%b%Y:%H:%M:%S'))

            # HRQOL base year from HRQOL dataset
            self.hrqol_0 = SimpleNamespace(df=pd.DataFrame([]), sql_handler=None,
                                           csv_path=join(cons.HRQOL_DATASET_FOLDER, cons.HRQOL_Year0_DATA_FILE),
                                           name='hrqol_0',
                                           encoding='utf-8',
                                           standard_time=('DateReceived', '%d-%m-%Y'))
            # HRQOL 1 year follow up from HRQOL dataset
            self.hrqol_1 = SimpleNamespace(df=pd.DataFrame([]), sql_handler=None,
                                           csv_path=join(cons.HRQOL_DATASET_FOLDER, cons.HRQOL_Year1_DATA_FILE),
                                           name='hrqol_1',
                                           encoding='utf-8',
                                           standard_time=('DateReceived', '%Y-%m-%d'))
            # HRQOL 3 year follow up from HRQOL dataset
            self.hrqol_3 = SimpleNamespace(df=pd.DataFrame([]), sql_handler=None,
                                           csv_path=join(cons.HRQOL_DATASET_FOLDER, cons.HRQOL_Year3_DATA_FILE),
                                           name='hrqol_3',
                                           encoding='utf-8',
                                           standard_time=('DateReceived', '%Y-%m-%d'))
            # HRQOL 5 year follow up from HRQOL dataset
            self.hrqol_5 = SimpleNamespace(df=pd.DataFrame([]), sql_handler=None,
                                           csv_path=join(cons.HRQOL_DATASET_FOLDER, cons.HRQOL_Year5_DATA_FILE),
                                           name='hrqol_5',
                                           encoding='utf-8',
                                           standard_time=('DateReceived', '%Y-%m-%d'))

    def open_csv_files(self, df_names='all'):
        """
        This function opens the csv files in the APPROACH dataset and saves them as pandas DataFrames
        (Just if the CSV path is provided)
        :param df_names: name of the dataframe to open (string or a list of strings). If 'all' is provided,
        all the dataframes will be opened
        :return:
        """

        if df_names == 'all':
            df_list = [table for table in self.data.__dict__.keys()]
        elif isinstance(df_names, str):
            df_list = [df_names]
        elif isinstance(df_names, list):
            df_list = df_names
        else:
            raise ValueError('df_name should be a string or a list of strings')
        # open the csv files
        for table_name in df_list:
            if self.data.__dict__[table_name].csv_path is not None:
                self.data.__dict__[table_name].df = pd.read_csv(self.data.__dict__[table_name].csv_path,
                                                                dtype=str,
                                                                encoding=self.data.__dict__[table_name].encoding)
                self.standardize_column_names(df_names=table_name)
                self.standardize_procedure_time(df_names=table_name)
                print(f'{table_name} is opened')
            else:
                self.data.__dict__[table_name].df = pd.DataFrame([])
                print(f'{table_name} is not opened')

    def close_csv_files(self, df_names='all'):
        """
        This function closes the csv files in the APPROACH dataset and saves them as pandas DataFrames
        (Just if the CSV path is provided)
        :param df_names: name of the dataframe to open (string or a list of strings). If 'all' is provided,
        all the dataframes will be opened
        :return:
        """
        if df_names == 'all':
            df_list = [table for table in self.data.__dict__.keys()]
        elif isinstance(df_names, str):
            df_list = [df_names]
        elif isinstance(df_names, list):
            df_list = df_names
        else:
            raise ValueError('df_name should be a string or a list of strings')
        # close the csv files
        for table_name in df_list:
            self.data.__dict__[table_name].df = pd.DataFrame([])
            print(f'{table_name} is closed')

    def retrieve_patient(self, identifier_num, table_name, identifier_type='File No'):
        """
        This method returns a patient's data in a certain table
        :param identifier_num: patient's unique identifier as an integer number
        :param table_name: name of the table
        :param identifier_type: patient identifier's standard name in columns (default: File No) (you can use procedure number too)
        :return: a DataFrame of the procedures of the patient in the table
        """
        # get procedures with the file_no
        if self.data.__dict__[table_name].sql_handler is None:
            df_procedures = self.data.__dict__[table_name].df.loc[
                self.data.__dict__[table_name].df[identifier_type] == identifier_num]
        else:
            sql_handler = self.data.__dict__[table_name].sql_handler  # type:ApproachSqlDataset
            query = f'SELECT * FROM {sql_handler.schema_name}."{sql_handler.table_name}" WHERE "{sql_handler.patient_id_column}" = {identifier_num}'
            self.data.__dict__[table_name].df = sql_handler.execute_query(query)
            self.standardize_column_names(table_name)
            self.standardize_procedure_time(table_name)

            df_procedures = self.data.__dict__[table_name].df

        return df_procedures

    def standardize_column_names(self, df_names='all'):
        """
        This method standardize names across different databases (like common names)
        :param df_names: name of the dataframe to open (string or a list of strings). If 'all' is provided,
        all the dataframes will be opened
        :return:
        """
        # get a list of table names to standardize
        if df_names == 'all':
            df_list = [table for table in self.data.__dict__.keys()]
        elif isinstance(df_names, str):
            df_list = [df_names]
        elif isinstance(df_names, list):
            df_list = df_names
        else:
            raise ValueError('df_names should be a string or a list of strings')

        # change file_no
        file_no_standard_name = 'File No'
        file_no_possible_names = ['File No', 'fileNO', 'fileNo', 'FILENO', 'FILE_NO', 'file_no']

        for df_name in df_list:

            if 'carat' in df_name:
                self.data.__dict__[df_name].df['Procedure Number'] = self.data.__dict__[df_name].df[
                    'Procedure Number'].astype(int)
                continue  # skip carat tables since it does not have file_no

            if self.data.__dict__[df_name].df.empty:
                continue

            # check the dataframes in Data class and if 'File No' is not standard, rename it
            for file_no_name in file_no_possible_names:
                if file_no_name in self.data.__dict__[df_name].df.columns:
                    self.data.__dict__[df_name].df.rename(columns={file_no_name: file_no_standard_name}, inplace=True)

            # change File No type to int
            self.data.__dict__[df_name].df[file_no_standard_name] = self.data.__dict__[df_name].df[
                file_no_standard_name].astype(int)
            if df_name in ['cath', 'pci', 'cabg']:
                self.data.__dict__[df_name].df['Procedure Number'] = self.data.__dict__[df_name].df[
                    'Procedure Number'].astype(int)

    def standardize_procedure_time(self, df_names='all'):
        """
        This function goes over all dataframes and makes a standard time column called 'Procedure Start Date' from the
        procedure dates and appends it to the dataframes.
        :return:
        """
        # get a list of table names to standardize
        if df_names == 'all':
            df_list = [table for table in self.data.__dict__.keys()]
        elif isinstance(df_names, str):
            df_list = [df_names]
        elif isinstance(df_names, list):
            df_list = df_names
        else:
            raise ValueError('df_names should be a string or a list of strings')

        time_col_standard_name = 'Procedure Standard Time'

        for df_name in df_list:
            if self.data.__dict__[df_name].df.empty:
                continue

            # get the standard time column name and format
            column_name, format_str = self.data.__dict__[df_name].standard_time

            if df_name == 'accs':
                visit_time = self.data.__dict__[df_name].df['VISTIME']  # type: pd.DataFrame
                visit_time.fillna('0000', inplace=True)  # fill NaN with 0000
                visit_time[visit_time.str.len() == 3] = "0" + visit_time[
                    visit_time.str.len() == 3]  # add a zero in front of time like 859 to make it 0859
                visit_time[visit_time.str.len() < 3] = "0000"  # if it has less than 3 digits, make it 0000

                visit_datetime_str = self.data.__dict__[df_name].df['VISDATE'].astype(str) + ':' + visit_time
                self.data.__dict__[df_name].df[time_col_standard_name] = pd.to_datetime(visit_datetime_str,
                                                                                        format='%Y%m%d:%H%M')
            else:  # for all other dataframes
                # check if the time column is already in datetime format
                if self.data.__dict__[df_name].df[column_name].dtype == 'datetime64[ns]':
                    self.data.__dict__[df_name].df[time_col_standard_name] = self.data.__dict__[df_name].df[
                        column_name]
                else:
                    # convert to datetime
                    self.data.__dict__[df_name].df[time_col_standard_name] = pd.to_datetime(
                        self.data.__dict__[df_name].df[column_name], format=format_str)

    def get_all_procedures(self, from_file=None):
        """
        This method runs over all patients and finds the corresponding Cath procedure for every other procedure
        WARNING: This function is time consuming
        :param from_file: if the function is already run and you have the results, just load it
        :return: changes self.all_procedures
        """
        if from_file is not None:
            # Just read the data from file (if it exists)
            self.all_procedures = pd.read_csv(from_file)
        else:
            for file_no in self.all_patients:
                patient = Patient(file_no, self)
                patient.procedures['File No'] = patient.file_no

                # if it is the first patient, define the procedures dataframe. Otherwise, just append
                if len(self.all_procedures.columns) == 0:
                    self.all_procedures = pd.concat([self.all_procedures, patient.procedures], axis=1)
                else:
                    self.all_procedures = pd.concat([self.all_procedures, patient.procedures], axis=0,
                                                    ignore_index=True)

        # print(self.all_procedures)

    def find_subsequent_revasc_for_each_cath(self):
        """
        This function finds the first treatment (revascularization) after each catheterization. It uses the cohort's
        procedures and finds first revasc. after each cath. and writes it is CATH dataframe under "SubsequentTreatment"
        column. If there is no revasc. after the cath., the treatment is medical therapy alone.
        :return:
        """
        pci_cabg_procedures = self.all_procedures[self.all_procedures['Procedure Table'].isin(['pci', 'cabg'])]
        first_pci_cabg_procedures = pci_cabg_procedures.groupby('Procedure Number of corr. Cath').first().reset_index()

        pci_only = first_pci_cabg_procedures[first_pci_cabg_procedures['Procedure Table'] == 'pci']
        cabg_only = first_pci_cabg_procedures[first_pci_cabg_procedures['Procedure Table'] == 'cabg']

        self.data.cath.df['SubsequentTreatment'] = 'Medical Therapy'  # initialization - default treatment is MT

        self.data.cath.df['SubsequentTreatment'].loc[
            self.data.cath.df['Procedure Number'].isin(pci_only['Procedure Number of corr. Cath'])] = 'PCI'

        self.data.cath.df['SubsequentTreatment'].loc[
            self.data.cath.df['Procedure Number'].isin(cabg_only['Procedure Number of corr. Cath'])] = 'CABG'

        # self.data.df_cath.to_csv('/Users/peyman/Desktop/temp2.csv')


class Patient:
    databases_analyzed = {}
    file_no = None
    birth_year = None
    cohort = None
    procedures = None
    outcomes = None

    def __init__(self, patient_file_no: int, cohort: Cohort, load_from_path=None):
        """
        This class is used to store all the information about a patient.
        It retrieves the information from the database and stores it in the class.
        :param patient_file_no:
        :param cohort:
        :param load_from_path:
        """

        if load_from_path is not None:
            self.load(load_from_path, cohort, verbose=True)  # load the patient from file
        else:
            self.file_no = patient_file_no
            self.cohort = cohort
            self.procedures = pd.DataFrame([],
                                           columns=['Procedure Number', 'Procedure Table', 'Procedure Standard Time',
                                                    'JsonInfo'])
            self.outcomes = self.Outcomes()

            # this dictionary is used to determine if a certain database has been analyzed for this patient
            self.databases_analyzed = {'vs': False, 'dad': False,
                                       'nacrs': False, 'accs': False,
                                       'claims': False, 'hrqol_0': False,
                                       'hrqol_1': False, 'hrqol_3': False,
                                       'hrqol_5': False, 'cath': False,
                                       'pci': False, 'cabg': False}

            self.get_revascularization_procedures()
            self.find_corresponding_cath_for_procedures()

    def save(self, path, verbose=False):
        """
        This method saves the patient object to a folder
        :param path: path to save the patient
        :param verbose: if True, prints the path
        :return:
        """
        # save DataFrames as csv files
        self.procedures.to_csv(os.path.join(path, 'procedures.csv'), index=False)
        for df_name in self.outcomes.__dict__:
            try:
                self.outcomes.__dict__[df_name].to_csv(os.path.join(path, df_name + '.csv'), index=False)
            except AttributeError as e:
                if verbose:
                    print(f'Patient {self.file_no} does not have {df_name} DataFrame: ' + str(e))

        # save the other attributes as json
        with open(os.path.join(path, 'attributes.json'), 'w') as f:
            json_data = {
                'file_no': self.file_no,
                'birth_year': self.birth_year,
                'databases_analyzed': self.databases_analyzed
            }
            json.dump(json_data, f)

        if verbose:
            print(f'Patient {self.file_no} object saved in: {path}')

    def load(self, path, cohort, verbose=False):
        """
        This method loads the patient object from a folder
        :param path: path to load the patient
        :param cohort: the cohort object that the patient belongs to (from Cohort class)
        :param verbose: if True, prints the path
        :return:
        """
        self.cohort = cohort

        # load DataFrames as csv files
        self.procedures = pd.read_csv(os.path.join(path, 'procedures.csv'))
        self.outcomes = self.Outcomes()
        for df_name in self.outcomes.__dict__:
            try:
                self.outcomes.__dict__[df_name] = pd.read_csv(os.path.join(path, df_name + '.csv'))
            except FileNotFoundError as e:
                print(f'This patient does not have {df_name} DataFrame: ' + str(e))

        # load the other attributes as json
        with open(os.path.join(path, 'attributes.json'), 'r') as f:
            json_data = json.load(f)
            self.file_no = json_data['file_no']
            self.birth_year = json_data['birth_year']
            self.databases_analyzed = json_data['databases_analyzed']

        if verbose:
            print(f'Patient {self.file_no} object loaded from: {path}')

    def fill_age(self):
        """
        This method fills the age column in the procedures DataFrame for all procedures (if it is not already filled)
        :return:
        """
        age_column = 'Age_at_cath'
        if age_column not in self.procedures.columns:
            pass
        else:
            self.procedures[age_column].fillna(
                self.procedures['Procedure Standard Time'].dt.year - self.birth_year, inplace=True)

    def fill_sex(self):
        """
        This method fills the sex column in the procedures DataFrame for all procedures (if it is not already filled)
        with any available sex information
        """
        sex_column = 'Sex_Female'
        if sex_column not in self.procedures.columns:
            pass
        else:
            self.procedures[sex_column].fillna(method='ffill', inplace=True)
            self.procedures[sex_column].fillna(method='bfill', inplace=True)

    def determine_checkup_times(self, time_after=None, time_before=None):
        """
        This method determines the checkup times for this patient. Checkup times are the times that the patient's
        data need to be analyzed. It will be every year starting from the first procedure date (or the time_after)
        until the last procedure date (or the time_before). If there is a catheterization procedure happens before
        the end of the checkup period, a new checkup time will be added for that day. the next checkup time will be
        based on the previous checkup time (a year after that). The ceckup times will be stored in the
        self.procedures DataFrame with "chekcup" in the Procedure Table column.
        :param time_after: the first checkup time will be after this date. pd.Timestamp object
        :param time_before: the last checkup time will be before this date pd.Timestamp object
        :return:
        """
        if time_after is None:
            time_after = self.procedures['Procedure Standard Time'].min()
        if time_before is None:
            time_before = self.procedures['Procedure Standard Time'].max()

        checkup_procedures = pd.DataFrame([],
                                          columns=['Procedure Number', 'Procedure Table', 'Procedure Standard Time'])

        # first checkup time
        checkup_procedures_count = 0
        checkup_procedures = pd.concat([checkup_procedures, pd.DataFrame(
            {'Procedure Number': [checkup_procedures_count], 'Procedure Table': ['checkup'],
             'Procedure Standard Time': [time_after]})], ignore_index=True)

        # add following checkup times
        for i, row in self.procedures.iterrows():
            if row['Procedure Standard Time'] > time_before:
                break
            if row['Procedure Standard Time'] > time_after:
                # if you reach a cath, add a checkup time for that day and store the cath number in the JsonInfo column
                if row['Procedure Table'] == 'cath':
                    checkup_procedures_count += 1
                    checkup_procedures = pd.concat([checkup_procedures, pd.DataFrame(
                        {'Procedure Number': [checkup_procedures_count], 'Procedure Table': ['checkup_cath'],
                         'Procedure Standard Time': [row['Procedure Standard Time']],
                         'JsonInfo': json.dumps({'CathNo': row['Procedure Number']})})], ignore_index=True)

                else:
                    # check every other procedure to see if a checkup time is needed
                    while checkup_procedures.iloc[-1]['Procedure Standard Time'] + pd.DateOffset(
                            days=cons.TIME_BETWEEN_TWO_CHECKUPS) < row['Procedure Standard Time']:
                        checkup_procedures_count += 1
                        checkup_procedures = pd.concat([checkup_procedures, pd.DataFrame(
                            {'Procedure Number': [checkup_procedures_count], 'Procedure Table': ['checkup'],
                             'Procedure Standard Time': [
                                 checkup_procedures.iloc[-1]['Procedure Standard Time'] + pd.DateOffset(
                                     days=cons.TIME_BETWEEN_TWO_CHECKUPS)]})], ignore_index=True)

        # change the time of all checkups end of the day
        checkup_procedures['Procedure Standard Time'] = checkup_procedures['Procedure Standard Time'].apply(
            lambda x: x.replace(hour=23, minute=59, second=59))

        self.add_to_patient_procedures(checkup_procedures)

    def aggregate_based_on_checkups(self, aggregation_strategy_df: pd.DataFrame, time_threshold=365):
        """
        This method aggregates the data in the procedures DataFrame based on the checkup times. The aggregation
        strategy is given as a DataFrame with the following columns:
        Variable_name,include_in_cath,include_in_checkup,aggregation_strategy,variable_type
        :param aggregation_strategy_df: a DataFrame with the mentioned columns
        :param time_threshold: the maximum time before a checkup that procedures can be included in that checkup
        for aggregation (in Days)
        :return: a DataFrame with the aggregated data
        """
        # create a dictionary of aggregation strategies for each variable
        agg_strategies = dict(
            zip(aggregation_strategy_df['Variable_name'], aggregation_strategy_df['aggregation_strategy']))

        # aggregate procedures for each checkup
        aggregated_df = pd.DataFrame()
        checkups = self.procedures[self.procedures['Procedure Table'].isin(['checkup', 'checkup_cath'])]
        for checkup_time in checkups['Procedure Standard Time'].tolist():
            date_threshold = checkup_time - pd.DateOffset(days=time_threshold)
            filtered_df = self.procedures[(self.procedures['Procedure Standard Time'] > date_threshold) & (
                    self.procedures['Procedure Standard Time'] <= checkup_time)]
            # Aggregating rows according to the specific strategy for each column
            agg_values = []
            selected_vars = []
            for col, strategy in agg_strategies.items():
                # if the column is not in the procedures table or all of its values are nan, add None
                if col not in filtered_df.columns:
                    agg_values.append(None)
                    selected_vars.append(col)
                    continue
                elif filtered_df[col].isna().all():
                    agg_values.append(None)
                    selected_vars.append(col)
                    continue

                if strategy in ['max', 'or']:
                    agg_values.append(filtered_df[col].max())
                    selected_vars.append(col)
                elif strategy == 'min':
                    agg_values.append(filtered_df[col].min())
                    selected_vars.append(col)
                elif strategy == 'mean':
                    agg_values.append(filtered_df[col].mean())
                    selected_vars.append(col)
                elif strategy == 'recent':
                    if not filtered_df.empty:
                        agg_values.append(filtered_df[col].iloc[-1])
                    else:
                        agg_values.append(None)
                    selected_vars.append(col)
                elif strategy == 'recent_cath':
                    # if the strategy is recent_cath, we need to filter the rows to only include cath rows
                    cath_filtered_df = filtered_df[filtered_df['Procedure Table'] == 'cath']
                    if not cath_filtered_df.empty:
                        agg_values.append(cath_filtered_df[col].iloc[-1])
                    else:
                        agg_values.append(None)
                    selected_vars.append(col)
                else:
                    # if the strategy is not defined, just put none for now
                    agg_values.append(None)
                    selected_vars.append(col)

            # Create a DataFrame row with the aggregated values
            agg_row = pd.DataFrame([agg_values], columns=selected_vars)
            aggregated_df = pd.concat([aggregated_df, agg_row], ignore_index=True)

        # impute missing binary variables for ICD and ATC codes
        # List of binary columns
        binary_cols_list = \
            aggregation_strategy_df[aggregation_strategy_df['variable_type'].isin(['icd_binary', 'atc_binary'])][
                'Variable_name'].tolist()

        # Looping over each binary column
        for col in binary_cols_list:
            # If the first row is NaN, replace it with 0
            if pd.isna(aggregated_df[col].iloc[0]):
                aggregated_df[col].iloc[0] = 0

            # Use forward fill for remaining missing values
            aggregated_df[col].fillna(method='ffill', inplace=True)

        return aggregated_df

    def get_death_time(self):
        """
        This method returns the time of death of the patient
        :return: a datetime object
        """
        # check if the patient has VS in the procedures table
        if 'vs' in self.procedures['Procedure Table'].tolist():
            # if yes, return the time of the last VS
            return self.procedures[self.procedures['Procedure Table'] == 'vs']['Procedure Standard Time'].iloc[-1]
        else:
            # if no, return None
            return None

    def add_to_patient_procedures(self, procedures):
        """
        This method puts a procedure into the patients procedures and sorts them
        the added procedure must have columns: 'Procedure Number', 'Procedure Table', 'Procedure Standard Time'
        :return:
        """
        self.procedures = pd.concat([self.procedures, procedures], ignore_index=True)
        self.procedures['Procedure Standard Time'] = pd.to_datetime(self.procedures['Procedure Standard Time'])
        self.procedures.replace({True: 1, False: 0}, inplace=True)  # replace True and False with 1 and 0
        self.procedures.sort_values(by='Procedure Standard Time', inplace=True, ignore_index=True)  # sort by date

    def get_revascularization_procedures(self):
        """
        This method looks into PCI and CABG dataframes in the Cohort object and finds matching File No of the patient and sorts
        the procedures in time order
        :return: Changes self.procedures attribute with a dataframe containing procedure number, date, and the table
        """
        revasc_procedures = pd.DataFrame([], columns=['Procedure Number', 'Procedure Table', 'Procedure Standard Time'])

        for df_name in ['cath', 'pci', 'cabg']:
            # get procedures with the file_no
            df_procedures = self.cohort.retrieve_patient(self.file_no, df_name)  # type: pd.DataFrame
            df_procedures['Procedure Table'] = df_name

            # # bring cath features
            if df_name == 'cath':
                revasc_procedures = pd.concat([revasc_procedures, df_procedures], ignore_index=True)

                # Subtract the 'age' column from the 'date' column to get the birth year
                self.birth_year = int((
                                              pd.to_datetime(revasc_procedures['Procedure Standard Time']).dt.year -
                                              revasc_procedures[
                                                  'Age_at_cath']).astype(int).iloc[0])
            else:
                revasc_procedures = pd.concat([revasc_procedures, df_procedures[
                    ['Procedure Number', 'Procedure Table', 'Procedure Standard Time']
                ]], ignore_index=True)

        revasc_procedures['Procedure Standard Time'] = pd.to_datetime(revasc_procedures['Procedure Standard Time'])

        # write it to patient's procedures
        self.add_to_patient_procedures(revasc_procedures)
        self.databases_analyzed['cath'] = True
        self.databases_analyzed['pci'] = True
        self.databases_analyzed['cabg'] = True

    def find_corresponding_cath_for_procedures(self):
        """
        This function will find corresponding Cath procedure for every other procedure
        :return:
        """
        df_cath_procedures = self.procedures[['Procedure Number', 'Procedure Standard Time']].loc[
            self.procedures['Procedure Table'] == 'cath']

        merged_df = pd.merge_asof(self.procedures, df_cath_procedures, suffixes=('', ' of corr. Cath'),
                                  on='Procedure Standard Time', direction='backward',
                                  tolerance=pd.Timedelta(cons.MAX_VALID_DAYS_BTWN_CATH_AND_PROCEDURE, 'd'))

        self.procedures = pd.concat([self.procedures, merged_df['Procedure Number of corr. Cath']], axis=1)

    def get_admin_vs_data(self):
        """
        This method processes Vital Statistics data (death time)
        :return:
        """
        vs_procedures = self.cohort.retrieve_patient(self.file_no, 'vs')  # type: pd.DataFrame
        if not vs_procedures.empty:
            vs_procedures['Procedure Number'] = vs_procedures.index
            vs_procedures.drop_duplicates(subset=['Procedure Standard Time', 'U_CAUSE'],
                                          inplace=True)  # there are repetitive records of death, remove duplicates
            vs_procedures['Procedure Table'] = 'vs'
            vs_procedures['JsonInfo'] = np.nan

            # add cardiovascular death cause to the Json data
            vs_procedures['IsCardiovascularDeath'] = vs_procedures['U_CAUSE'].apply(
                lambda x: icd.is_cardiovascular_death_icd10(x))
            vs_procedures['JsonInfo'].json.join(vs_procedures['IsCardiovascularDeath'].apply(
                lambda x: json.dumps({'IsCardiovascularDeath': x})), inplace=True)

            # change death hour to 23:59:59 so that other procdures do not go after that (it doesn't have hour normally)
            vs_procedures['Procedure Standard Time'] = vs_procedures['Procedure Standard Time'].apply(
                lambda dt: dt.replace(hour=23, minute=59, second=59))

            self.add_to_patient_procedures(
                vs_procedures[['Procedure Number',
                               'Procedure Table',
                               'Procedure Standard Time',
                               'JsonInfo',
                               'IsCardiovascularDeath']])

        self.databases_analyzed['vs'] = True

    @staticmethod
    def get_disease_from_admission_df(procedures: pd.DataFrame, disease_cols: list, result_col_name: str,
                                      icd_function) -> pd.DataFrame:
        """
        This method gets a dataframe from admission datasets (DAD, NACRS, ACCS) and checks if a certain disease is
        present for each row. You need to write a helper function that receives an ICD code as string and return true or
        false for that disease
        :param procedures: the dataframe (DAD, NACRS, ACCS)
        :param disease_cols: The column names of ICD codes as a list
        :param result_col_name: The column name in the result dataframe in which the presence of the disease will be returned as True or False
        :param icd_function: a helper function that receives an ICD code as string and return true or false for that disease
        :return: a Dataframe similar to the input datafrme with column named by {result_col_name} indicating the disease is present or not (True/False)
                and updates the JsonInfo column with the {result_col_name}
        """
        _procedures = procedures.copy(deep=True)
        _procedures[result_col_name] = False  # series indicating if procedure has the disease
        # do for each column of DXCODEs
        for dxcode in disease_cols:
            _procedures[dxcode] = _procedures[dxcode].astype(str)
            _procedures[result_col_name] = _procedures[result_col_name] | _procedures[dxcode].apply(
                lambda x: icd_function(x))  # if at least one column has the disease
            # _procedures['JsonInfo'].json.join(
            #     _procedures[result_col_name].apply(lambda x: json.dumps({result_col_name: x})),
            #     inplace=True)  # make it json

        return _procedures

    def get_admin_dad_data(self):
        """
        This method processes Discharge Abstract Data
        :return:
        """
        dad_procedures = self.cohort.retrieve_patient(self.file_no, 'dad')  # type: pd.DataFrame
        if not dad_procedures.empty:
            dad_procedures['Procedure Number'] = dad_procedures.index
            dad_procedures.drop_duplicates(subset=['Procedure Standard Time'],
                                           inplace=True)  # there are repetitive records, remove duplicates
            dad_procedures['Procedure Table'] = 'dad'  # assign table name
            dad_procedures['JsonInfo'] = np.nan

            # ---------
            # add Acute Myocardial Infarction (AMI) Events
            # ---------
            dad_procedures = self.get_disease_from_admission_df(dad_procedures,
                                                                disease_cols=cons.DAD_DISEASE_CODE_COLS,
                                                                result_col_name='HasAMI',
                                                                icd_function=icd.is_acute_myocardial_infarction)
            # ---------
            # add stroke events
            # ---------
            dad_procedures = self.get_disease_from_admission_df(dad_procedures,
                                                                disease_cols=cons.DAD_DISEASE_CODE_COLS,
                                                                result_col_name='HasStroke',
                                                                icd_function=icd.is_stroke)
            # ---------
            # add RIW cost
            # ---------
            dad_procedures['RIWCost'] = dad_procedures['RIW'].apply(lambda x: float(x))

            # ---------
            # add Selected ICD codes
            # ---------
            icd_codes_list = pd.read_csv(cons.DAD_SELECTED_ICD_FEATURES)
            for ref_icd_code in icd_codes_list['code']:
                dad_procedures = self.get_disease_from_admission_df(dad_procedures,
                                                                    disease_cols=cons.DAD_DISEASE_CODE_COLS,
                                                                    result_col_name=ref_icd_code,
                                                                    icd_function=lambda x:
                                                                    icd.is_icd_code_equal_to(x, ref_icd_code))

            self.add_to_patient_procedures(
                dad_procedures[
                    ['Procedure Number',
                     'Procedure Table',
                     'Procedure Standard Time',
                     'JsonInfo',
                     'HasAMI',
                     'HasStroke',
                     'RIWCost',
                     *icd_codes_list['code'].tolist()]])

        self.databases_analyzed['dad'] = True

    def get_admin_nacrs_data(self):
        """
        This method processes National Ambulatory Care Reporting System data
        :return:
        """
        nacrs_procedures = self.cohort.retrieve_patient(self.file_no, 'nacrs')  # type: pd.DataFrame
        if not nacrs_procedures.empty:
            nacrs_procedures['Procedure Number'] = nacrs_procedures.index
            nacrs_procedures.drop_duplicates(subset=['Procedure Standard Time'],
                                             inplace=True)  # there are repetitive records, remove duplicates
            nacrs_procedures['Procedure Table'] = 'nacrs'  # assign table name
            nacrs_procedures['JsonInfo'] = np.nan

            # ---------
            # add Acute Myocardial Infarction (AMI) Events
            # ---------
            nacrs_procedures = self.get_disease_from_admission_df(nacrs_procedures,
                                                                  disease_cols=cons.NACRS_DISEASE_CODE_COLS,
                                                                  result_col_name='HasAMI',
                                                                  icd_function=icd.is_acute_myocardial_infarction)
            # ---------
            # add stroke events
            # ---------
            nacrs_procedures = self.get_disease_from_admission_df(nacrs_procedures,
                                                                  disease_cols=cons.NACRS_DISEASE_CODE_COLS,
                                                                  result_col_name='HasStroke',
                                                                  icd_function=icd.is_stroke)

            # ---------
            # add RIW cost
            # ---------
            nacrs_procedures['RIWCost'] = nacrs_procedures['CACS_RIW'].apply(lambda x: float(x))

            # ---------
            # add Selected ICD codes
            # ---------
            icd_codes_list = pd.read_csv(cons.NACRS_SELECTED_ICD_FEATURES)
            for ref_icd_code in icd_codes_list['code']:
                nacrs_procedures = self.get_disease_from_admission_df(nacrs_procedures,
                                                                      disease_cols=cons.NACRS_DISEASE_CODE_COLS,
                                                                      result_col_name=ref_icd_code,
                                                                      icd_function=lambda x:
                                                                      icd.is_icd_code_equal_to(x, ref_icd_code))

            self.add_to_patient_procedures(
                nacrs_procedures[
                    ['Procedure Number',
                     'Procedure Table',
                     'Procedure Standard Time',
                     'JsonInfo',
                     'HasAMI',
                     'HasStroke',
                     'RIWCost',
                     *icd_codes_list['code'].tolist()]])

        self.databases_analyzed['nacrs'] = True

    def get_admin_accs_data(self):
        """
        This method processes ACCS data
        :return:
        """
        accs_procedures = self.cohort.retrieve_patient(self.file_no, 'accs')  # type: pd.DataFrame
        if not accs_procedures.empty:
            accs_procedures['Procedure Number'] = accs_procedures.index
            accs_procedures.drop_duplicates(subset=['Procedure Standard Time'],
                                            inplace=True)  # there are repetitive records, remove duplicates
            accs_procedures['Procedure Table'] = 'accs'  # assign table name
            accs_procedures['JsonInfo'] = np.nan

            # ---------
            # add Acute Myocardial Infarction (AMI) Events
            # ---------
            accs_procedures = self.get_disease_from_admission_df(accs_procedures,
                                                                 disease_cols=cons.ACCS_DISEASE_CODE_COLS,
                                                                 result_col_name='HasAMI',
                                                                 icd_function=icd.is_acute_myocardial_infarction)
            # ---------
            # add stroke events
            # ---------
            accs_procedures = self.get_disease_from_admission_df(accs_procedures,
                                                                 disease_cols=cons.ACCS_DISEASE_CODE_COLS,
                                                                 result_col_name='HasStroke',
                                                                 icd_function=icd.is_stroke)

            # ---------
            # add RIW cost
            # ---------
            # TODO: ACCS does not have RIW cost, they might be a way to estimate it using the procedure codes
            accs_procedures['RIWCost'] = np.nan

            self.add_to_patient_procedures(
                accs_procedures[
                    ['Procedure Number',
                     'Procedure Table',
                     'Procedure Standard Time',
                     'JsonInfo',
                     'HasAMI',
                     'HasStroke']])

        self.databases_analyzed['accs'] = True

    def get_admin_claims_data(self):
        """
        This method processes claims data
        :return:
        """
        claims_procedures = self.cohort.retrieve_patient(self.file_no, 'claims')  # type: pd.DataFrame
        if not claims_procedures.empty:
            claims_procedures['Procedure Number'] = claims_procedures.index
            claims_procedures['Procedure Table'] = 'claims'  # assign table name
            claims_procedures['JsonInfo'] = np.nan
            claims_procedures['Proc Cost Dollars'] = claims_procedures['COST_USE_RS'].apply(lambda x: float(x))

            self.add_to_patient_procedures(
                claims_procedures[
                    ['Procedure Number',
                     'Procedure Table',
                     'Procedure Standard Time',
                     'Proc Cost Dollars',
                     'JsonInfo']])

        self.databases_analyzed['claims'] = True

    def get_admin_lab_data(self):
        """
        This method processes lab data - This lab data is a modified version of the original lab data
        to select only those lab tests that are relevant to the study (top 100 lab tests)
        :return:
        """
        lab_procedures_raw = self.cohort.retrieve_patient(self.file_no, 'lab')  # type: pd.DataFrame
        lab_cleaner = LabDataCleaningHelper(patient_lab_df=lab_procedures_raw)
        lab_procedures = lab_cleaner.make_feature_space()
        if not lab_procedures.empty:
            lab_procedures['Procedure Number'] = np.arange(0, len(lab_procedures))
            lab_procedures['Procedure Standard Time'] = lab_procedures.index
            lab_procedures['Procedure Table'] = 'lab'
            lab_procedures['JsonInfo'] = np.nan

            lab_procedures.reset_index(inplace=True, drop=True)  # reset index to be able to add to the database

            self.add_to_patient_procedures(lab_procedures)

        self.databases_analyzed['lab'] = True

    def get_admin_pin_data(self):
        """
        This method processes PIN data (pharmaceutical information network). It will use a list of selected drug codes
        (ATC codes) and one-hot-encode the prescription of that drug for each patient
        :return:
        """
        pin_procedures = self.cohort.retrieve_patient(self.file_no, 'pin')
        if not pin_procedures.empty:
            # ---------
            # add drug codes
            # if the ATC code starts with the given code, then it is considered as prescribed for the patient
            # ---------
            pin_selected_drugs = pd.read_csv(cons.PIN_SELECTED_ATC_FEATURES, usecols=['code'])['code'].to_list()

            temp_df_list = [pin_procedures]
            for atc_code in pin_selected_drugs:
                drug_name = 'ATC_' + atc_code
                temp_df = pd.DataFrame()
                temp_df[drug_name] = pin_procedures['SUPP_DRUG_ATC_CODE'].apply(
                    lambda x: x.startswith(atc_code) if pd.notnull(x) else False)
                temp_df_list.append(temp_df)

            pin_procedures = pd.concat(temp_df_list, axis=1)  # connect the columns to the main DataFrame

            # aggregate the drug codes in the same day
            pin_procedures = pin_procedures[['Procedure Standard Time',
                                             *['ATC_' + atc_code for atc_code in pin_selected_drugs]]]
            pin_procedures['Procedure Standard Time'] = pd.to_datetime(
                pin_procedures['Procedure Standard Time']).dt.date
            pin_procedures = pin_procedures.groupby('Procedure Standard Time').max()

            # add other columns
            pin_procedures['Procedure Standard Time'] = pin_procedures.index
            pin_procedures.reset_index(inplace=True, drop=True)
            pin_procedures['Procedure Number'] = np.arange(0, len(pin_procedures))
            pin_procedures['Procedure Table'] = 'pin'

            self.add_to_patient_procedures(pin_procedures)

        self.databases_analyzed['pin'] = True

    def get_hrqol_data(self):
        # TODO: HRQOL questionnaire data are recieved in times far away from the the date sent (e.g., 234883). Check
        #  which one is better as standard time
        """
        This method processes HRQOL data for each patient
        :return:
        """
        hrqol_dataframes = [
            'hrqol_0',
            'hrqol_1',
            'hrqol_3',
            'hrqol_5']
        # Initialize an empty DataFrame to store the filtered rows
        hrqol_procedures = pd.DataFrame()
        for df_name in hrqol_dataframes:
            df_filtered = self.cohort.retrieve_patient(self.file_no, df_name)  # type: pd.DataFrame
            if not df_filtered.empty:
                df_filtered['Procedure Number'] = df_filtered.index
                df_filtered['Procedure Table'] = df_name  # assign table name
                df_filtered['EQ5D Utility Score'] = df_filtered[cons.HRQOL_EQ5D_COLS].apply(
                    lambda row: summarize_EQ5D(*[float(row[c]) for c in cons.HRQOL_EQ5D_COLS]), axis=1)

                saq_cols = list(cons.HRQOL_SAQ_COLS.keys()) if isinstance(cons.HRQOL_SAQ_COLS,
                                                                          dict) else cons.HRQOL_SAQ_COLS
                df_filtered['SAQ Utility Score'] = df_filtered[saq_cols].apply(
                    lambda row: summarize_SAQ({c: float(row[c]) for c in saq_cols}), axis=1)

                hrqol_procedures = pd.concat([hrqol_procedures, df_filtered], ignore_index=True)
            else:
                continue
        if not hrqol_procedures.empty:
            self.add_to_patient_procedures(
                hrqol_procedures[['Procedure Number',
                                  'Procedure Table',
                                  'Procedure Standard Time',
                                  'EQ5D Utility Score',
                                  'SAQ Utility Score']])

        self.databases_analyzed['hrqol_0'] = True
        self.databases_analyzed['hrqol_1'] = True
        self.databases_analyzed['hrqol_3'] = True
        self.databases_analyzed['hrqol_5'] = True

    def find_subsequent_revasc_for_each_cath(self):
        """
        This function finds the first treatment (revascularization) after each catheterization. It uses the cohort's
        procedures and finds first revasc. after each cath. and writes it is CATH dataframe under "SubsequentTreatment"
        column. If there is no revasc. after the cath., the treatment is medical therapy alone.
        :return:
        """
        pci_cabg_procedures = self.procedures[self.procedures['Procedure Table'].isin(['pci', 'cabg'])]
        first_pci_cabg_procedures = pci_cabg_procedures.groupby('Procedure Number of corr. Cath').first().reset_index()

        pci_only = first_pci_cabg_procedures[first_pci_cabg_procedures['Procedure Table'] == 'pci']
        cabg_only = first_pci_cabg_procedures[first_pci_cabg_procedures['Procedure Table'] == 'cabg']

        self.procedures.loc[self.procedures[
                                'Procedure Table'] == 'cath', 'SubsequentTreatment'] = 'Medical Therapy'  # initialization - default treatment is MT

        self.procedures['SubsequentTreatment'].loc[
            (self.procedures['Procedure Number'].isin(pci_only['Procedure Number of corr. Cath'])) &
            (self.procedures['Procedure Table'] == 'cath')] = 'PCI'

        self.procedures['SubsequentTreatment'].loc[
            (self.procedures['Procedure Number'].isin(cabg_only['Procedure Number of corr. Cath'])) &
            (self.procedures['Procedure Table'] == 'cath')] = 'CABG'

    class Outcomes:
        def __init__(self):
            self.repeated_revasc = None  # repeated revascularization for each cath.
            self.mace = None  # 3-point composite major adverse cardiovascular event
            self.survival = None  # 90-day survival for each cath.
            self.hrqol_saq = None  # Health-Related Quality of Life - SAQ questionnaire
            self.hrqol_eq5d = None  # Health-Related Quality of Life - EQ5D questionnaire
            self.cost_dollar = None  # cost of treatment - dollar values
            self.cost_riw = None  # cost of treatment - Resource Intensity Weight
            self.cost_utility = None  # Cost-Utility ratio for treatments in different subgroups

    def get_outcome_quality_of_life(self):
        # TODO: you need to calculate the change in quality of life for each catheterization
        # This will can be done by substracting two consecutive HRQOL questionnaires for each catheterization
        """
        This method finds the quality of life for each catheterization
        :return:
        """
        # check if required tables are analyzed
        required_tables = ['cath', 'vs', 'hrqol_0', 'hrqol_1', 'hrqol_3', 'hrqol_5']
        if not all([self.databases_analyzed[table] for table in required_tables]):
            raise ValueError(f'Missing required tables for HRQOL outcome calculation: {required_tables}')

        followup_time = cons.OUTCOME_HRQOL_YEARS * 365  # days

        hrqol = pd.DataFrame([], columns=['Cath No', 'SAQ', 'EQ5D'])
        hrqol['Cath No'] = self.procedures['Procedure Number'].loc[
            self.procedures['Procedure Table'] == 'cath'].reset_index(drop=True)

        # pull death details from VS
        death_details = self.procedures.loc[self.procedures['Procedure Table'] == 'vs'].reset_index(drop=True)
        if len(death_details):
            death_time = death_details['Procedure Standard Time'][0]
        else:
            death_time = np.nan

        for cath_no in hrqol['Cath No']:
            cath_procedures = self.procedures.loc[self.procedures['Procedure Number'] == cath_no].reset_index(drop=True)
            cath_time = cath_procedures['Procedure Standard Time'][0]

            # find the closest HRQOL questionnaire
            hrqol_procedures = self.procedures.loc[(self.procedures['Procedure Table'].str.contains('hrqol')) &
                                                   (self.procedures['Procedure Standard Time'] >= cath_time) &
                                                   (self.procedures[
                                                        'Procedure Standard Time'] <= cath_time + pd.Timedelta(
                                                       followup_time, 'day'))
                                                   ].reset_index(drop=True)
            if len(hrqol_procedures) > 1:
                years_between_two_hrqol = (hrqol_procedures['Procedure Standard Time'][1] - hrqol_procedures[
                    'Procedure Standard Time'][0]).days / 365
                if years_between_two_hrqol < 2:
                    hrqol['SAQ'][hrqol['Cath No'] == cath_no] = (hrqol_procedures['SAQ Utility Score'][1] -
                                                                 hrqol_procedures['SAQ Utility Score'][0])

                    hrqol['EQ5D'][hrqol['Cath No'] == cath_no] = (hrqol_procedures['EQ5D Utility Score'][1] -
                                                                  hrqol_procedures['EQ5D Utility Score'][0])

            elif len(hrqol_procedures) == 1:
                if death_time is not np.nan:
                    years_between_two_hrqol = (death_time - hrqol_procedures['Procedure Standard Time'][0]).days / 365
                    if years_between_two_hrqol < 2:
                        hrqol['SAQ'][hrqol['Cath No'] == cath_no] = hrqol_procedures['SAQ Utility Score'][0]
                        hrqol['EQ5D'][hrqol['Cath No'] == cath_no] = hrqol_procedures['EQ5D Utility Score'][0]

        self.outcomes.hrqol_saq = hrqol[['Cath No', 'SAQ']]
        self.outcomes.hrqol_eq5d = hrqol[['Cath No', 'EQ5D']]

    def get_outcome_cost_riw(self):
        # TODO: not implemented yet
        """
        This method finds the cost of treatment for each catheterization in Resource Intensity Weight (RIW)
        Cath No: Cath Identifier
        RIW Cost: Accumulated cost of treatment for this cath
        :return:
        """
        return None

    def get_outcome_cost_dollar(self):
        """
        This method finds the cost of treatment for each catheterization in dollar values
        Cath No: Cath Identifier
        Dollar Cost: Accumulated cost of treatment for this cath
        :return:
        """
        # check if required tables are analyzed
        required_tables = ['cath', 'vs', 'claims']
        if not all([self.databases_analyzed[table] for table in required_tables]):
            raise ValueError(f'Missing required tables for Cost outcome calculation: {required_tables}')

        checkup_events = ['checkup', 'checkup_cath']

        # TODO: is there any other outcome to be divided to the cost? like MACE.
        followup_time = cons.OUTCOME_COST_DOLLAR_YEARS * 365  # days
        cost_dollar = pd.DataFrame([], columns=['checkup No', 'Dollar Cost', 'Time Earned (days)'])

        # pull death details from VS
        death_details = self.procedures.loc[self.procedures['Procedure Table'] == 'vs'].reset_index(drop=True)
        if len(death_details):
            death_time = death_details['Procedure Standard Time'][0]
            death_time = pd.to_datetime(death_time)
        else:
            death_time = np.nan

        # assign all cath numbers to the cost table
        checkup_df = self.procedures.loc[self.procedures['Procedure Table'].isin(checkup_events)].reset_index(drop=True)
        cost_dollar['checkup No'] = checkup_df['Procedure Number']

        for checkup_no in cost_dollar['checkup No']:
            this_checkup = checkup_df.loc[checkup_df['Procedure Number'] == checkup_no].reset_index(drop=True)
            # if checkup is a cath, find the cath number associated to it
            if this_checkup['Procedure Table'].values[0] == 'checkup_cath':
                cath_no = json.loads(this_checkup['JsonInfo'].values[0])
                cath_no = cath_no['CathNo']
                # find the last treatment associated to this cath. (Last PCI/CABG, or the Cath itself if nothing done)
                treatments_associated_to_cath = self.procedures[
                    self.procedures['Procedure Number of corr. Cath'] == cath_no].reset_index(drop=True)
                last_treatment_time = max(treatments_associated_to_cath['Procedure Standard Time'])
            else:  # if it is simple checkup, last treatment is the checkup itself (medical therapy)
                last_treatment_time = this_checkup['Procedure Standard Time'].values[0]
            last_treatment_time = pd.to_datetime(last_treatment_time)

            # find the claims associated with this cath within the followup time
            claims = self.procedures.loc[(self.procedures['Procedure Table'] == 'claims') &
                                         (self.procedures['Procedure Standard Time'] >= last_treatment_time) &
                                         (self.procedures[
                                              'Procedure Standard Time'] <= last_treatment_time + pd.Timedelta(
                                             followup_time, 'day'))
                                         ].reset_index(drop=True)
            if claims.empty:
                cost_dollar.loc[cost_dollar['checkup No'] == checkup_no, 'Dollar Cost'] = 0
                cost_dollar.loc[cost_dollar['checkup No'] == checkup_no, 'Time Earned (days)'] = followup_time
            else:

                cost_dollar.loc[cost_dollar['checkup No'] == checkup_no, 'Dollar Cost'] = claims['Proc Cost Dollars'].sum()

                # find the time earned for this cath (min of followup time and time to death)
                if isinstance(death_time, float):
                    if np.isnan(death_time):
                        time_earned = followup_time
                else:
                    # print(f"death time: {type(death_time)}, last treatment time: {type(last_treatment_time)}, followup time: {type(followup_time)}")
                    time_earned = min(followup_time, (death_time - last_treatment_time).days)
                cost_dollar.loc[cost_dollar['checkup No'] == checkup_no, 'Time Earned (days)'] = time_earned

        self.outcomes.cost_dollar = cost_dollar

    def get_outcome_repeated_revasc(self):
        """
        This method finds the time to repeated revascularization for each catheterization
        Cath No: Cath Identifier
        Event No: Repeated Revascularization following the Cath (not the treatment for this cath)
        Event type: The table of the revascularization (PCI/CABG)
        Time to Event: How many days from the last treatment associated to the Cath to the Repeated Revascularization?

        :return: change self.outcomes.repeated_revasc
        """
        # check if required tables are analyzed
        required_tables = ['cath', 'pci', 'cabg']
        if not all([self.databases_analyzed[table] for table in required_tables]):
            raise ValueError(f'Missing required tables for Repeated Revasc. outcome calculation: {required_tables}')

        checkup_events = ['checkup', 'checkup_cath']

        repeated_revasc = pd.DataFrame([], columns=['checkup No', 'Event No', 'Event Type', 'Time to Event (days)'])

        # assign all cath numbers to the revasc table
        checkup_df = self.procedures.loc[self.procedures['Procedure Table'].isin(checkup_events)].reset_index(drop=True)
        repeated_revasc['checkup No'] = checkup_df['Procedure Number']

        for checkup_no in repeated_revasc['checkup No']:
            this_checkup = checkup_df.loc[checkup_df['Procedure Number'] == checkup_no].reset_index(drop=True)
            # if checkup is a cath, find the cath number associated to it
            if this_checkup['Procedure Table'].values[0] == 'checkup_cath':
                cath_no = json.loads(this_checkup['JsonInfo'].values[0])
                cath_no = cath_no['CathNo']
                # find the last treatment associated to this cath. (Last PCI/CABG, or the Cath itself if nothing done)
                treatments_associated_to_cath = self.procedures[
                    self.procedures['Procedure Number of corr. Cath'] == cath_no].reset_index(drop=True)
                last_treatment_time = max(treatments_associated_to_cath['Procedure Standard Time'])
            else:  # if it is simple checkup, last treatment is the checkup itself (medical therapy)
                last_treatment_time = this_checkup['Procedure Standard Time'].values[0]
                cath_no = np.nan  # no cath associated to this checkup

            # find the checkup time
            checkup_time = this_checkup['Procedure Standard Time'].values[0]

            # find next revascularization procedure (pci/cabg) which is not associated with this checkup
            next_repeated_revasc = self.procedures.loc[
                (self.procedures['Procedure Standard Time'] > checkup_time) &
                (self.procedures['Procedure Number of corr. Cath'] != cath_no) &
                (self.procedures['Procedure Table'].isin(['pci', 'cabg']))
                ].reset_index(drop=True)

            # if there are any repeated cath, store the first one
            if len(next_repeated_revasc):
                repeated_revasc['Event No'].loc[repeated_revasc['checkup No'] == checkup_no] = next_repeated_revasc[
                    'Procedure Number'][0]
                repeated_revasc['Event Type'].loc[repeated_revasc['checkup No'] == checkup_no] = next_repeated_revasc[
                    'Procedure Table'][0]
                # calc time to revascularization
                repeated_revasc['Time to Event (days)'].loc[repeated_revasc['checkup No'] == checkup_no] = (
                        pd.to_datetime(next_repeated_revasc['Procedure Standard Time']) - last_treatment_time).dt.days[
                    0]

        self.outcomes.repeated_revasc = repeated_revasc
        # print(self.outcomes.repeated_revasc)

    def get_outcome_mace(self):
        """
        This method will calculate the time to Major Adverse Cardiovascular Events (MACE)
        MACE definition:
            - acute myocardial infarction (AMI)
            - stroke
            - cardiovascular death
        :return: changes self.outcomes.mace
        """
        # check if required tables are analyzed
        required_tables = ['cath', 'dad', 'nacrs', 'vs']
        if not all([self.databases_analyzed[table] for table in required_tables]):
            raise ValueError(f'Missing required tables for MACE outcome calculation: {required_tables}')

        checkup_events = ['checkup', 'checkup_cath']

        mace = pd.DataFrame([], columns=['checkup No', 'Event No', 'Event Type', 'Time to Event (days)'])
        # assign all cath numbers to the survival table
        checkup_df = self.procedures.loc[self.procedures['Procedure Table'].isin(checkup_events)].reset_index(drop=True)
        mace['checkup No'] = checkup_df['Procedure Number']

        # store first mace for each cath
        for checkup_no in mace['checkup No']:
            this_checkup = checkup_df.loc[checkup_df['Procedure Number'] == checkup_no].reset_index(drop=True)
            # find the last treatment associated to this cath. (Last PCI/CABG, or the Cath itself if nothing done)
            if this_checkup['Procedure Table'].values[0] == 'checkup_cath':
                cath_no = json.loads(this_checkup['JsonInfo'].values[0])
                cath_no = cath_no['CathNo']
                # find the last treatment associated to this cath. (Last PCI/CABG, or the Cath itself if nothing done)
                treatments_associated_to_cath = self.procedures[
                    self.procedures['Procedure Number of corr. Cath'] == cath_no].reset_index(drop=True)
                last_treatment_time = max(treatments_associated_to_cath['Procedure Standard Time'])
            else:  # if it is simple checkup, last treatment is the checkup itself (medical therapy)
                last_treatment_time = this_checkup['Procedure Standard Time'].values[0]
            # find maces after each cath
            # mace_details = self.procedures.loc[
            #     ((self.procedures['HasAMI']==1) | (self.procedures['HasStroke']==1) | (self.procedures['IsCardiovascularDeath']==1)) &
            #     (self.procedures['Procedure Standard Time'] > last_treatment_time)
            #     ].reset_index(drop=True)
            mace_details = self.procedures.loc[
                ((self.procedures.get('HasAMI', 0)==1 ) |
                 (self.procedures.get('HasStroke', 0)==1) |
                 (self.procedures.get('IsCardiovascularDeath', 0)==1)) &
                (self.procedures['Procedure Standard Time'] > last_treatment_time)
                ].reset_index(drop=True)

            if len(mace_details):
                patient_has_mace = True
                mace_time = mace_details['Procedure Standard Time'][0]
            else:
                patient_has_mace = False
                mace_time = np.nan

            if patient_has_mace:
                mace['Event No'][mace['checkup No'] == checkup_no] = mace_details['Procedure Number'][0]
                mace['Event Type'][mace['checkup No'] == checkup_no] = mace_details['Procedure Table'][0]
                mace['Time to Event (days)'][mace['checkup No'] == checkup_no] = (
                        mace_time - last_treatment_time).days
            else:
                mace['Event No'][mace['checkup No'] == checkup_no] = np.nan
                mace['Event Type'][mace['checkup No'] == checkup_no] = np.nan
                mace['Time to Event (days)'][mace['checkup No'] == checkup_no] = np.nan

        self.outcomes.mace = mace

    def get_outcome_survival(self):
        """
        This method will find the survival for each catheterization
        :return: Changes self.outcome.survival
        """
        # check if required tables are analyzed
        required_tables = ['cath', 'vs']
        if not all([self.databases_analyzed[table] for table in required_tables]):
            raise ValueError(f'Missing required tables for survival outcome calculation: {required_tables}')

        checkup_events = ['checkup', 'checkup_cath']

        survival = pd.DataFrame([], columns=['checkup No', 'Event No', 'Event Type', 'Time to Event (days)'])
        # assign all cath numbers to the survival table
        checkup_df = self.procedures.loc[self.procedures['Procedure Table'].isin(checkup_events)].reset_index(drop=True)
        survival['checkup No'] = checkup_df['Procedure Number']

        # pull death details from VS
        death_details = self.procedures.loc[self.procedures['Procedure Table'] == 'vs'].reset_index(drop=True)
        if len(death_details):
            patient_has_died = True
            death_time = death_details['Procedure Standard Time'][0]
        else:
            patient_has_died = False
            death_time = np.nan

        for checkup_no in survival['checkup No']:
            this_checkup = checkup_df.loc[checkup_df['Procedure Number'] == checkup_no].reset_index(drop=True)
            # if checkup is a cath, find the cath number associated to it
            if this_checkup['Procedure Table'].values[0] == 'checkup_cath':
                cath_no = json.loads(this_checkup['JsonInfo'].values[0])
                cath_no = cath_no['CathNo']
                # find the last treatment associated to this cath. (Last PCI/CABG, or the Cath itself if nothing done)
                treatments_associated_to_cath = self.procedures[
                    self.procedures['Procedure Number of corr. Cath'] == cath_no].reset_index(drop=True)
                last_treatment_time = max(treatments_associated_to_cath['Procedure Standard Time'])
            else:  # if it is simple checkup, last treatment is the checkup itself (medical therapy)
                last_treatment_time = this_checkup['Procedure Standard Time'].values[0]

            if patient_has_died:
                survival['Event No'][survival['checkup No'] == checkup_no] = death_details['Procedure Number'][0]
                survival['Event Type'][survival['checkup No'] == checkup_no] = death_details['Procedure Table'][0]
                survival['Time to Event (days)'][survival['checkup No'] == checkup_no] = (
                        death_time - last_treatment_time).days
            else:
                survival['Event No'][survival['checkup No'] == checkup_no] = np.nan
                survival['Event Type'][survival['checkup No'] == checkup_no] = np.nan
                survival['Time to Event (days)'][survival['checkup No'] == checkup_no] = np.nan

        self.outcomes.survival = survival


def main():
    start_time_cohort = time.time()
    cohort = Cohort()
    # cohort.standardize_column_names()
    # cohort.standardize_procedure_time()
    end_time_cohort = time.time()
    elapsed_time_cohort = end_time_cohort - start_time_cohort
    print('Elapsed time for loading the cohort: ' + str(elapsed_time_cohort))

    # # Get all procedures and their correspondence
    # cohort.get_all_procedures(from_file='/Users/peyman/Desktop/all_procedures.csv')
    # cohort.all_procedures.to_csv('/Users/peyman/Desktop/all_procedures.csv')

    # cohort.find_subsequent_revasc_for_each_cath()

    start_time_patient = time.time()
    patient = Patient(224873, cohort=cohort)
    # print(patient.procedures)
    patient.get_admin_vs_data()
    patient.get_admin_dad_data()
    patient.get_admin_nacrs_data()
    patient.get_admin_accs_data()
    patient.get_hrqol_data()
    patient.get_admin_claims_data()
    patient.get_admin_lab_data()
    patient.get_admin_pin_data()

    patient.get_outcome_survival()
    patient.get_outcome_repeated_revasc()
    patient.get_outcome_cost_dollar()
    patient.get_outcome_mace()
    print(patient.outcomes.mace)
    print(patient.outcomes.cost_dollar)
    # print(patient.outcomes.hrqol_saq)
    # print(patient.outcomes.hrqol_eq5d)

    patient.procedures.to_csv(os.path.join(cons.ROOT_FOLDER, 'patient_procedures_sample.csv'))
    end_time_patient = time.time()
    elapsed_time_patient = end_time_patient - start_time_patient
    print('Elapsed time for calculating patient outcomes: ' + str(elapsed_time_patient))
    """
    Interesting cases:
    109776,
    """


if __name__ == '__main__':
    main()
