import os

REMOTE = True  # True if program is running on a remote server, otherwise False

if REMOTE:
    ROOT_FOLDER = '/home/peyman.ghasemi1/data'
else:
    ROOT_FOLDER = '/Users/peyman/Documents/PhD _ Biomedical Engineering/Research/APPROACH Dataset'

# root folder of the this code
code_root = os.path.dirname(os.path.abspath(__file__))

# postgresql credentials
SQL_CREDENTIALS = os.path.join(code_root, 'sql-credentials.json')

#  Dataset name
DATASET_FOLDER = os.path.join(ROOT_FOLDER, 'APPROACH_V4_deID')
CATH_DATA_FILE = 'cath_v4_deID.csv'
PCI_DATA_FILE = 'pci_v4_deID.csv'
CABG_DATA_FILE = 'cabg_v4_deID.csv'
CARAT_MAIN_DATA_FILE = 'carat_main_v4_deID.csv'

"""
HRQOL Dataset
"""
HRQOL_DATASET_FOLDER = os.path.join(ROOT_FOLDER, 'APPROACH_HRQOL')
HRQOL_Year0_DATA_FILE = 'APPROACH_HRQOL_v1.csv'
HRQOL_Year1_DATA_FILE = 'Year1.csv'
HRQOL_Year3_DATA_FILE = 'Year3.csv'
HRQOL_Year5_DATA_FILE = 'Year5.csv'

HRQOL_EQ5D_COLS = ['EUROQOL_MOBILITY', 'EUROQOL_SELFCARE', 'EUROQOL_USUAL', 'EUROQOL_PAIN', 'EUROQOL_ANXIETY']
HRQOL_SAQ_COLS = {'SAQ_Dressing': 6, 'SAQ_Walking': 6, 'SAQ_Showering': 6, 'SAQ_Climbing': 6, 'SAQ_Gardening': 6,
                  'SAQ_WalkBrisk': 6, 'SAQ_Running': 6, 'SAQ_Lifting': 6, 'SAQ_Strenuous': 6,
                  'SAQ42': 5, 'SAQ43': 6, 'SAQ44': 6, 'SAQ45': 6, 'SAQ46': 5, 'SAQ47': 5, 'SAQ48': 5,
                  'SAQ49': 5, 'SAQ410': 5, 'SAQ411': 5}


"""
ADMIN Dataset
"""
ADMIN_DATASET_FOLDER = os.path.join(ROOT_FOLDER, 'ADMIN')
# VS
ADMIN_VS_FILE = 'VS_v2.csv'
# DAD
ADMIN_DAD_FILE = 'DAD_v2.csv'
DAD_DISEASE_CODE_COLS = ['DXCODE' + str(i) for i in range(1, 26)]
# NACRS
ADMIN_NACRS_FILE = 'NACRS_v2.csv'
NACRS_DISEASE_CODE_COLS = ['DXCODE' + str(i) for i in range(1, 11)]
# ACCS
ADMIN_ACCS_FILE = 'ACCS_v2.csv'
ACCS_DISEASE_CODE_COLS = ['DXCODE' + str(i) for i in range(1, 11)]
# CLAIMS
ADMIN_CLAIMS_FILE = 'CLAIMS_v2.csv'

"""
Variables
"""
VARIABLES_JSON = os.path.join(code_root, 'variables.json')
DAD_SELECTED_ICD_FEATURES = os.path.join(code_root, 'selected_features/DAD/concrete_with_weights.csv')
NACRS_SELECTED_ICD_FEATURES = os.path.join(code_root, 'selected_features/NACRS/concrete_with_weights.csv')
PIN_SELECTED_ATC_FEATURES = os.path.join(code_root, 'selected_features/PIN/concrete_with_weights.csv')
LAB_SELECTED_FEATURES = os.path.join(code_root, 'selected_features/LAB/number_of_patients_per_test.csv')

# Constant Values

# how many days between cath and a procedure (PCI, CABG,...)
# is valid to consider that cath corresponded to the procedure?
MAX_VALID_DAYS_BTWN_CATH_AND_PROCEDURE = 90

# how many days between two checkups in a normal situation?
TIME_BETWEEN_TWO_CHECKUPS = 365


# Outcomes time horizon (followup until these times)
OUTCOME_SURVIVAL_FOLLOWUP_DAYS = 90
OUTCOME_REPEATED_REVASC_YEARS = 3
OUTCOME_MACE_YEARS = 3
OUTCOME_COST_DOLLAR_YEARS = 3
OUTCOME_COST_RIW_YEARS = 3
OUTCOME_HRQOL_YEARS = 5
OUTCOME_COST_UTILITY_YEARS = 3
