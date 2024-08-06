import pandas as pd
import tabulate
import numpy as np


def concat_dummies(df, col_name, drop_original=True, drop_classes=None):
    """
    This function concatenates the dummy columns of a categorical feature
    :param df: DataFrame
    :param col_name: str
    :param drop_original: bool (whether to drop the original column or not)
    :param drop_classes: list of str (classes to be dropped)
    :return: DataFrame
    """
    if drop_classes is None:
        drop_classes = []

    df_dummies = pd.get_dummies(df[col_name], prefix=col_name)
    df = pd.concat([df, df_dummies], axis=1)

    if drop_original:
        df.drop(col_name, axis=1, inplace=True)

    # if unwanted classes are specified ant they are present in the df, drop them
    for drop_class in drop_classes:
        df.drop(col_name + '_' + drop_class, axis=1, inplace=True, errors='ignore')

    return df


def calc_time_since_prior_event(cath_df, prior_col_name, cath_time_col_name):
    """
    This function calculates the time since the prior event
    :param cath_df: DataFrame (cath)
    :param prior_col_name: str
    :param cath_time_col_name: str (cath time)
    :return: DataFrame (time since the prior event until the cath)
    """
    prior_dates_str = cath_df[prior_col_name].fillna('')
    prior_dates = prior_dates_str.str.split(',')
    prior_dates = prior_dates.apply(
        lambda x: [pd.to_datetime(date, errors='coerce') if date != '' else pd.NaT for date in x])
    latest_prior_date = prior_dates.apply(lambda x: max(x) if len(x) > 0 else pd.NaT)
    time_since_prior = pd.to_datetime(cath_df[cath_time_col_name]) - latest_prior_date
    time_since_prior.fillna(pd.Timedelta(days=365 * 100), inplace=True)  # fill NaNs with 100 years
    time_since_prior = time_since_prior.apply(lambda x: x.days)

    return time_since_prior


def clean_cath(cath, table_name='cath'):
    """
    This function cleans the cath dataset
    :param cath: DataFrame - APPROACH cath dataset (raw)
    :param table_name: str (name of the table)
    :return:
    """
    selected_features = pd.DataFrame()

    selected_features['File No'] = cath['File No'].astype(int)
    selected_features['Procedure Number'] = cath['Procedure Number'].astype(int)
    selected_features['Procedure Table'] = table_name
    selected_features['Procedure Standard Time'] = pd.to_datetime(cath['Procedure Standard Time'])
    selected_features['Age_at_cath'] = cath['Age_at_cath'].astype(float)

    selected_features['Sex'] = cath['Sex']
    selected_features = concat_dummies(selected_features, 'Sex', drop_original=True, drop_classes=['Male'])

    selected_features['Facility'] = cath['Facility']
    selected_features = concat_dummies(selected_features, 'Facility', drop_original=True)

    selected_features['heightcm'] = cath['heightcm'].astype(float)
    selected_features['weightkg'] = cath['weightkg'].astype(float)

    selected_features['CCS'] = cath['CCS']
    selected_features['CCS'].fillna('Not Entered', inplace=True)
    selected_features = concat_dummies(selected_features, 'CCS', drop_original=True)

    selected_features['Priority'] = cath['Priority']
    selected_features = concat_dummies(selected_features, 'Priority', drop_original=True, drop_classes=['Unknown'])

    # handle procedure_completed
    pc_col_name = 'procedures_completed'
    selected_features[pc_col_name] = cath['procedures_completed']
    selected_features[pc_col_name].fillna('', inplace=True)
    procedure_completed_list = [
        'Diagnostic: Coronary Angiogram',
        'LV Angiogram',
        'Left Heart Cath',
        'Graft Angiogram',
        'Right Heart Catheterization',
        'Radial Angiogram',
        'Adjunct: Pressure Wire Measurements',
        'Aortic Root',
        'Iliac/Femoral Angiogram',
        'Cardiac Output-Thermal'
    ]
    for proc in procedure_completed_list:
        selected_features[pc_col_name + '_' + proc] = selected_features[pc_col_name].str.contains(proc).astype(int)
    selected_features.drop(pc_col_name, axis=1, inplace=True)

    # handle indication type
    indication_col_name = 'indicationtyp'
    selected_features[indication_col_name] = cath['indicationtyp']
    indications_list = ['Acute Coronary Syndrome',
                        'Stable Angina',
                        'Valvular Heart Disease',
                        'Other',
                        'Congestive Heart Failure']
    # Change values to 'other' if not in indications_list
    selected_features[indication_col_name] = np.where(
        selected_features[indication_col_name].isin(indications_list),
        selected_features[indication_col_name], 'other')
    selected_features = concat_dummies(selected_features, indication_col_name, drop_original=True)

    # handle indicationdx
    indicationdx_col_name = 'indicationdx'
    selected_features[indicationdx_col_name] = cath['indicationdx']
    selected_features[indicationdx_col_name].loc[selected_features[indicationdx_col_name] == 'Unknown'] = 'Not Entered'
    selected_features = concat_dummies(selected_features, indicationdx_col_name, drop_original=True)

    # handle symptoms
    symptoms_col_name = 'symptoms'
    selected_features[symptoms_col_name] = cath['symptoms']
    selected_features[symptoms_col_name].fillna('Not Entered', inplace=True)
    selected_features = concat_dummies(selected_features, symptoms_col_name, drop_original=True)

    # handle smoking
    selected_features['Smoking'] = cath['Smoking']
    selected_features = concat_dummies(selected_features, 'Smoking', drop_original=True)

    # handle diabetes
    selected_features['Diabetes'] = cath['diabtype']
    selected_features['Diabetes'].loc[cath['diabetes'] == 'N'] = 'No'  # define no diabetes class
    selected_features['Diabetes'].fillna('Type II', inplace=True)  # the remaining NaNs are type II
    selected_features = concat_dummies(selected_features, 'Diabetes', drop_original=True)

    # handle diabetes therapy
    selected_features['diabtherapy'] = cath['diabtherapy']
    selected_features['diabtherapy'].loc[selected_features['diabtherapy'].isin(['Diet', 'None', 'Other'])] = 'Other'
    selected_features = concat_dummies(selected_features, 'diabtherapy', drop_original=True)

    # handle yes/no data
    yes_no_list = [
        'pre_shock',
        'pre_cardiac_arrest',
        'pre_chf',
        'pre_intubate',
        'pre_inotrope',
        'pre_IABP',
        'pre_none',
        'syncope',
        'dyslipedimia',
        'HF',
        'HF2wks',
        'Afib',
        'pulmonaryhten',
        'hypertension',
        'Angina',
        'FamHx',
        'IE',
        'PE',
        'Radiation',
        'SleepApnea',
        'HomeO2',
        'CLD',
        'CEV',
        'Delirium',
        'Psych',
        'CEVD',
        'Renal',
        'CKD',
        'AKI',
        'Dialysis',
        'PAD',
        'Venous',
        'Varicose',
        'VeinStripping',
        'DVT',
        'PUD',
        'Liver',
        'Hypothyroid',
        'Hyperthyroid',
        'Metabolic',
        'Immuno',
        'Steroid',
        'rheumatoid',
        'coagulopathy',
        'malignancy',
        'druguse',
        'HIV',
        'Alcohol',
        'ASA'
    ]
    for col in yes_no_list:
        selected_features[col] = cath[col]
        selected_features[col].fillna('N', inplace=True)
        selected_features[col] = selected_features[col].replace({'Y': 1, 'N': 0})

    # handle Extent of CAD
    selected_features['ExtentCAD'] = cath['ExtentCAD']
    selected_features = concat_dummies(selected_features, 'ExtentCAD', drop_original=True)

    # handle prior Cath, PCI, CABG
    cath_time_col = 'Procedure Standard Time'
    prior_pci_col = 'Prior PCI Dates'
    prior_cabg_col = 'Prior CABG Dates'
    prior_cath_col = 'Prior CATH Dates'
    prior_mi_col = 'PriorMIDates'
    selected_features['days_since_prior_cath'] = calc_time_since_prior_event(cath, prior_cath_col, cath_time_col)
    selected_features['time_since_prior_pci'] = calc_time_since_prior_event(cath, prior_pci_col, cath_time_col)
    selected_features['time_since_prior_cabg'] = calc_time_since_prior_event(cath, prior_cabg_col, cath_time_col)
    selected_features['time_since_prior_mi'] = calc_time_since_prior_event(cath, prior_mi_col, cath_time_col)

    return selected_features


def clean_carat_main(carat):
    """
    Clean the carat data
    :param carat:
    :return:
    """
    selected_features = pd.DataFrame()
    selected_features['Procedure Number'] = carat['Procedure Number'].astype(int)

    selected_features['Dominance'] = carat['Dominance']
    selected_features = concat_dummies(selected_features, 'Dominance', drop_original=True)

    selected_features['dukeJeopardy'] = pd.to_numeric(carat['dukeJeopardy'], errors='coerce')

    selected_features['approachJeopardy'] = pd.to_numeric(carat['approachJeopardy'], errors='coerce')

    selected_features['dukeCoronaryIndex'] = pd.to_numeric(carat['dukeCoronaryIndex'], errors='coerce')

    for i in range(1, 18):
        selected_features['lvs' + str(i)] = pd.to_numeric(carat['lvs' + str(i)], errors='coerce')

    return selected_features


def connect_cath_and_carat(cath, carat):
    """
    Connect cath and carat data by procedure number
    :param cath: Dataframe of cath data
    :param carat: Dataframe of carat data
    :return: merged dataframe
    """
    merged_df = cath.merge(carat, how='left', on='Procedure Number')

    return merged_df


