import sys

import os
import numpy as np
import json
import time
import datetime
import pickle
import pandas as pd
from multiprocessing import Process, Manager, Pool
import logging
import training_constants as tc
from APPROACH.APPROACH import Cohort, Patient

logging.basicConfig(filename='my_log.log',  # Set the log file name
                    level=logging.DEBUG,     # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

start_time_cohort = time.time()
logging.info(f'Start loading the cohort at {datetime.datetime.now()}')

cohort = Cohort()

# open additional data from csv files
cohort.data.lab.sql_handler = None
cohort.data.pin.sql_handler = None
cohort.data.claims.sql_handler = None
cohort.data.lab.csv_path = '/home/peyman.ghasemi1/data/ADMIN/LAB_v2.csv'
cohort.open_csv_files(df_names='lab')
cohort.data.pin.csv_path = '/home/peyman.ghasemi1/data/ADMIN/PIN_v2.csv'
cohort.open_csv_files(df_names='pin')
cohort.data.claims.csv_path = '/home/peyman.ghasemi1/data/ADMIN/CLAIMS_v2.csv'
cohort.open_csv_files(df_names='claims')


# cohort.standardize_column_names()
# cohort.standardize_procedure_time()
end_time_cohort = time.time()
elapsed_time_cohort = end_time_cohort - start_time_cohort
logging.info(f'End loading the cohort at {end_time_cohort} and it took {elapsed_time_cohort} seconds')

# # TODO: remove this after testing
# sample_patients = pd.read_csv('sample_patients.csv')
# sample_patients['Processed'] = False
# sample_patients.to_csv('sample_patients.csv', index=False)
#############################################################



def process_patient(patient_file_no, processed_file_nos_queue, agg_sterategy_df, patient_list_csv, major_save_dir, cohort=cohort):

    try:

        subfolder = os.path.join(major_save_dir, str(patient_file_no // 1000))
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        save_dir = os.path.join(subfolder, str(patient_file_no))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        start_time_patient = time.time()
        patient = Patient(patient_file_no, cohort=cohort)

        patient.get_admin_vs_data()


        patient.get_admin_dad_data()


        patient.get_admin_nacrs_data()


        # patient.get_admin_accs_data()


        patient.get_hrqol_data()



        patient.get_admin_claims_data()



        patient.get_admin_lab_data()


        patient.get_admin_pin_data()


        patient.fill_age()
        patient.fill_sex()

        # find actions of each cath
        patient.find_subsequent_revasc_for_each_cath()

        # determine checkup times
        time_before = None
        patient.determine_checkup_times(time_after=pd.to_datetime('2008-01-01'))

        # Outcomes
        patient.get_outcome_survival()
        patient.get_outcome_repeated_revasc()
        patient.get_outcome_cost_dollar()
        patient.get_outcome_mace()

        end_time_patient = time.time()
        elapsed_time_patient = end_time_patient - start_time_patient
        
        # save the patient
        patient.save(save_dir, verbose=False)

        # aggregate the data for each checkup
        aggregated_df = patient.aggregate_based_on_checkups(agg_sterategy_df, time_threshold=365)

        # get patient's death time (if any) and add it to the aggregated_df as the last checkup
        patient_death_time = patient.get_death_time()
        if patient_death_time is not None:
            # if the patient died within the followup time, add the death time as the last checkup
            if patient_death_time - aggregated_df['Procedure Standard Time'].max() < datetime.timedelta(days=tc.death_followup_time):
                last_procedure_number = aggregated_df['Procedure Number'].max()
                patient_death_checkup = pd.DataFrame({'Procedure Standard Time': [patient_death_time], 'Procedure Table': ['death'], 'Procedure Number': [last_procedure_number + 1]})
                aggregated_df = pd.concat([aggregated_df, patient_death_checkup], ignore_index=True)
    
        # save the aggregated_df
        agg_path = os.path.join(save_dir, 'aggregated_df.csv')
        aggregated_df.to_csv(agg_path, index=False)

        # update the queue to show that this patient has been processed
        process_status = {'File No': patient_file_no, 'Processed': True}
        processed_file_nos_queue.put(process_status)
    except Exception as e:
        logging.error(f'Error in processing patient {patient_file_no} at {datetime.datetime.now()}: {e}')
        


def csv_writer(processed_file_nos_queue, patient_list_csv='sample_patients.csv'):
    """
    Write processed file nos to csv file
    :param processed_file_nos_queue: Queue of processed file nos - a dictionary with keys 'File No' and 'Processed'
    :param patient_list_csv: path to the csv file to write to
    """
    while True:
        # wait for a patient to be processed
        result = processed_file_nos_queue.get()
        if result == 'STOP':
            break
        patient_file_no = result['File No']
        # update the csv file to show that this patient has been processed
        sample_patients = pd.read_csv(patient_list_csv)
        sample_patients.loc[sample_patients['File No'] == patient_file_no, 'Processed'] = True
        sample_patients.to_csv(patient_list_csv, index=False)
        
        # print(f'Patient {patient_file_no} has been processed and the csv file has been updated.')




# we will save every file no in a separate folder inside folders named the integer division of file no by 1000
# e.g. file no 1234 will be saved in folder 1
# the reason for this is to avoid having too many files in one folder (it may cause problems in some file systems)
major_save_dir = 'all_patients_data'
if not os.path.exists(major_save_dir):
    os.makedirs(major_save_dir)

agg_sterategy_df = pd.read_csv('agg_strategy_v2.csv')

patient_list_csv = 'patiens_list.csv'
sample_patients = pd.read_csv(patient_list_csv)

# # TODO: remove this after testing
sample_patients['Processed'] = False
sample_patients.to_csv(patient_list_csv, index=False)

# create folders
for patient_file_no in sample_patients['File No'].to_list():
    subfolder = os.path.join(major_save_dir, str(patient_file_no // 1000))
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    save_dir = os.path.join(subfolder, str(patient_file_no))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# get unprocessed patients from csv file
unprocessed_patients = sample_patients[sample_patients['Processed'] == False]
patient_file_no_list = unprocessed_patients['File No'].to_list()

# Start writer process to write processed file nos to csv file
manager = Manager()
q = manager.Queue()
writer_process = Process(target=csv_writer, args=(q, patient_list_csv))
writer_process.start()

# Initialize worker pool
with Pool(processes=90) as pool:
    pool.starmap(process_patient, [(pf, q, agg_sterategy_df, patient_list_csv, major_save_dir) for pf in patient_file_no_list])

# Stop writer process
q.put('STOP')
writer_process.join()

logging.info(f"All tasks are done at {datetime.datetime.now()}")
print("All tasks are done!")
