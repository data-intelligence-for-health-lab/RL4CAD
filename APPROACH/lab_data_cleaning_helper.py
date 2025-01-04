import pandas as pd
import numpy as np
import os
import re
import APPROACH.constants as cons


class LabDataCleaningHelper:
    selected_lab_items = None
    lab_df = None
    lab_df_cleaned = None

    def __init__(self, patient_lab_df):
        self.selected_lab_items = get_lab_items()
        self.lab_df = self.get_selected_lab_data(patient_lab_df)

    def make_feature_space(self):
        """
        This function creates the feature space for the lab data
        It is essentially a dataframe with the index of the date of the lab test and the columns of the selected
        lab items and a flag for each test determining whether they are an approximate or real value.
        The values are the lab results.
        :return: a dataframe with the feature space for the lab data (the index is the date of the lab test)
        """
        # create a dataframe with the feature space
        feature_space_names = self.selected_lab_items + [f + '_APPROX' for f in self.selected_lab_items]
        feature_space = pd.DataFrame(columns=feature_space_names + ['Procedure Standard Time'])

        for item in self.selected_lab_items:
            item_df = self.process_lab_data(item)
            if item_df.empty:
                continue

            rows = []
            for _, row in item_df.iterrows():
                df_dict = {key: np.nan for key in self.selected_lab_items}
                df_dict[item] = row['TEST_RSLT_num']
                df_dict[item + '_APPROX'] = row['IS_APPRXMT']
                df_dict['Procedure Standard Time'] = row['Procedure Standard Time']
                rows.append(df_dict)

            feature_space = pd.concat([feature_space, pd.DataFrame.from_records(rows)], ignore_index=True)
            feature_space['Procedure Standard Time'] = pd.to_datetime(
                feature_space['Procedure Standard Time']).dt.normalize()

        # aggregate features happened on the same day
        agg_functions = {f: 'mean' for f in self.selected_lab_items}
        agg_functions.update({f + '_APPROX': 'max' for f in self.selected_lab_items})

        feature_space = feature_space.groupby('Procedure Standard Time')[feature_space_names].agg(agg_functions)

        return feature_space

    def get_selected_lab_data(self, lab_df):
        """
        This function separates the lab tests that are selected as features for our model
        :param lab_df:
        :return: df with only the selected lab tests
        """
        lab_df = lab_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)  # convert all to lowercase
        lab_df = lab_df[lab_df['TEST_NM'].isin(self.selected_lab_items)]
        return lab_df

    def process_lab_data(self, test_name):
        """
        This is a comprehensive function that cleans the lab data for a specific test
        It mainly takes care of the tests with special formatting or different units of measurement or wrong values
        It uses two helper functions: results_to_numeric and handle_urine_test_strips
        results_to_numeric is a general function that converts the results to numeric values (it handles spaces,
         <, >, notes, etc.)
        handle_urine_test_strips is a function that handles the urine test strips which have a special format of
        +1, +2, etc. It converts them to an approximate numeric value (in string) and passes them to results_to_numeric.
        :param test_name: string, the name of the test
        :return: a dataframe with the cleaned data for the test
        It raise an error if the test name is not in the list of selected lab tests
        """
        df = self.lab_df
        # if df is empty, return an empty dataframe
        if df.empty:
            return df

        # change the test names to the standard names
        change_test_names_dict = {
            "agap.": "anion gap",
            "red cell distribution width": "rdw",
            "creatine kinase": "ck",
            "mean corpuscular hgb conc": "mchc",
            "triglyceride": "triglycerides",
            "trig.": "triglycerides",
            "tsh": "thyroid stimulating hormone",
            "activated ptt": "ptt",
            "total iron binding capacity": "tibc",
            "iron binding capacity": "tibc",
            "nucleated red blood cell": "nrbc",
            "albumin/creatinine ratio,urine": "albumin creatinine ratio",
            "albumin/creatinine ratio,timed": "albumin creatinine ratio",
            "glucose fasting.": "glucose fasting",
            "glucose, fasting": "glucose fasting",
            "glucose,fasting (gestational)": "glucose fasting",
            "inr.": "inr",
            "inr raw": "inr",
            "ptt.": "ptt",
            "ptt (actin fs)": "ptt",
            "ptt actin fs": "ptt",
            "glucose - random": "glucose random",
            "glucose, random": "glucose random",
            "platelet count.": "platelet count",
            "neutrophil auto": "neut",
            "instr neut": "neut",
            "neutrophil#": "neut",
            "neutrophils": "neut",
            "m neutrophils": "neut",
            "neutrophil": "neut",
            "neutrophil #": "neut",
            "neutrophil total#": "neut",
            "instr nrbc": "nrbc",
            "nrbc auto": "nrbc",
            "p na": "sodium",
            "p k": "potassium",
            "p cl": "chloride",
            "hemoglobin a1c.": "hemoglobin a1c",
            "hemoglobin a1c (hba1c)": "hemoglobin a1c",
            "rdw cv": "rdw",
            "rdw-cv%": "rdw",
            "platelet raw": "platelet count",
            "tco2": "carbon dioxide",
            "hgb": "hemoglobin"
        }
        for name, target_name in change_test_names_dict.items():
            df["TEST_NM"] = df["TEST_NM"].replace(name, target_name)

        cols_to_keep = ['File No', 'Procedure Standard Time', 'TEST_NM', 'TEST_RSLT', 'TEST_UOFM',
                        'TEST_REF_RNG']
        date_col = 'TEST_VRFY_DTTM'
        # df[date_col] = pd.to_datetime(df[date_col], format='%d%b%Y:%H:%M:%S')

        if test_name == 'gfr-new':
            # get createnines
            gfr_df = df[df['TEST_NM'] == 'creatinine'][cols_to_keep + ['CLNT_AGE', 'CLNT_GNDR']]
            gfr_df['TEST_RSLT_num'], _ = results_to_numeric(gfr_df['TEST_RSLT'])

            age_df = pd.to_numeric(gfr_df['CLNT_AGE'], errors='coerce')
            k = np.where(gfr_df['CLNT_GNDR'] == 'female', 0.7, 0.9)
            alpha = np.where(gfr_df['CLNT_GNDR'] == 'female', -0.329, -0.411)
            is_female = np.where(gfr_df['CLNT_GNDR'] == 'female', 1.0, 0.0)

            # convert the values with unit of umol/L to mg/L (mg/L are wrongfully written as mmol/l)
            creatinine_molecular_weight = 113.12  # g/mol
            gfr_df.loc[gfr_df['TEST_UOFM'] == 'umol/l', 'TEST_RSLT_num'] *= (creatinine_molecular_weight / 1000.0)

            # convert to mg/dL
            gfr_df['TEST_RSLT_num'] /= 10.0
            gfr_df['TEST_UOFM'] = 'mg/dl'

            # claculate GFR
            gfr_df['gfr'] = 141 * (np.minimum(gfr_df['TEST_RSLT_num'].to_numpy() / k, 1) ** alpha) * (
                    np.maximum(gfr_df['TEST_RSLT_num'].to_numpy() / k, 1) ** -1.209) * (
                                    0.993 ** age_df.to_numpy()) * (1.018 ** is_female)
            test_df = gfr_df[cols_to_keep]
            test_df['TEST_NM'] = 'gfr-new'
            test_df['TEST_RSLT'] = gfr_df['gfr']
            test_df['TEST_UOFM'] = 'ml/min/1.73m2'
            test_df['TEST_REF_RNG'] = '-'

        else:
            test_df = df[df['TEST_NM'] == test_name.lower()][cols_to_keep]

        if test_name == 'creatinine':
            valid_units = ['umol/l']
            test_df['TEST_RSLT_num'], test_df['IS_APPRXMT'] = results_to_numeric(test_df['TEST_RSLT'])

            # convert the values with unit of mg/L to umol/L (they're wrongfully are written as mmol/l)
            creatinine_molecular_weight = 113.12  # g/mol
            test_df.loc[test_df['TEST_UOFM'] != 'umol/l', 'TEST_RSLT_num'] *= (1000.0 / creatinine_molecular_weight)
            test_df['TEST_UOFM'] = 'umol/l'

        elif test_name == 'hemoglobin':
            valid_units = ['g/l']
            test_df = test_df[test_df['TEST_UOFM'].isin(valid_units)]
            test_df['TEST_RSLT_num'], test_df['IS_APPRXMT'] = results_to_numeric(test_df['TEST_RSLT'])

        elif test_name == 'glomerular filtration rate estimate':
            test_df['TEST_UOFM'] = 'ml/min/1.73m2'  # all units are the same thing
            test_df['TEST_RSLT_num'], test_df['IS_APPRXMT'] = results_to_numeric(test_df['TEST_RSLT'])

        elif test_name == 'cholesterol':
            valid_units = ['mmol/l']
            test_df = test_df[test_df['TEST_UOFM'].isin(valid_units)]
            test_df['TEST_RSLT_num'], test_df['IS_APPRXMT'] = results_to_numeric(test_df['TEST_RSLT'])

        elif test_name == 'glucose ua':
            gluc_stript_dict = {
                **dict.fromkeys(['negative', 'neg', 'norm'], 0),
                **dict.fromkeys(['trace'], 5.5),
                **dict.fromkeys(['+1', '1+'], 14),
                **dict.fromkeys(['+2', '2+'], 28),
                **dict.fromkeys(['+3', '3+'], 55),
                **dict.fromkeys(['+4', '4+'], 110)
            }
            test_df['TEST_UOFM'] = 'mmol/l'
            test_df['TEST_RSLT_num'], test_df['IS_APPRXMT'] = handle_urine_test_strips(test_df['TEST_RSLT'],
                                                                                       gluc_stript_dict)

        elif test_name == 'ketones ua':
            ketones_stript_dict = {
                **dict.fromkeys(['negative', 'neg', 'normal', 'norm'], 0),
                **dict.fromkeys(['trace'], 0.5),
                **dict.fromkeys(['+1', '1+'], 1.5),
                **dict.fromkeys(['+2', '2+'], 3.9),
                **dict.fromkeys(['+3', '3+'], 10),
                **dict.fromkeys(['+4', '4+'], 20)
            }
            test_df['TEST_UOFM'] = 'mmol/l'
            test_df['TEST_RSLT_num'], test_df['IS_APPRXMT'] = handle_urine_test_strips(test_df['TEST_RSLT'],
                                                                                       ketones_stript_dict)

        elif test_name == 'protein urine ua':
            prot_stript_dict = {
                **dict.fromkeys(['negative', 'neg', 'normal', 'norm'], 0),
                **dict.fromkeys(['trace'], 0.15),
                **dict.fromkeys(['+1', '1+'], 0.3),
                **dict.fromkeys(['+2', '2+'], 1),
                **dict.fromkeys(['+3', '3+'], 3),
                **dict.fromkeys(['+4', '4+'], 10)
            }
            test_df['TEST_RSLT'].loc[
                test_df['TEST_RSLT'].str.contains('protein may be falsely elevated', na=False)] = 'NaN'
            test_df['TEST_UOFM'] = 'mmol/l'
            test_df['TEST_RSLT_num'], test_df['IS_APPRXMT'] = handle_urine_test_strips(test_df['TEST_RSLT'],
                                                                                       prot_stript_dict)

        elif test_name == 'blood ua':
            bld_stript_dict = {
                **dict.fromkeys(['negative', 'neg', 'normal', 'norm'], 0),
                **dict.fromkeys(['trace', 'trace-intact', 'trace-lysed', '0.3 (trace)'], 5),
                **dict.fromkeys(['small', '1+', '+1', '0.6 (1+)', '1.0 (1+)'], 10),
                **dict.fromkeys(['moderate', '2+', '+2', '2.0 (2+)', '5.0 (2+)'], 50),
                **dict.fromkeys(['large', '3+', '+3', '10.0 (heavy)'], 250),
                **dict.fromkeys(['4+', '5+', '>250', '>10.0 (heavy)'], 350)
            }

            test_df['TEST_UOFM'] = 'cells/ul'
            test_df['TEST_RSLT_num'], test_df['IS_APPRXMT'] = handle_urine_test_strips(test_df['TEST_RSLT'],
                                                                                       bld_stript_dict)

        elif test_name == 'leukocyte esterase ua':
            leu_stript_dict = {
                **dict.fromkeys(['negative', 'neg', 'normal', 'norm'], 0),
                **dict.fromkeys(['trace'], 15),
                **dict.fromkeys(['small', '1+', '+1'], 75),
                **dict.fromkeys(['moderate', '2+', '+2'], 125),
                **dict.fromkeys(['large', '3+', '+3'], 500),
                **dict.fromkeys(['4+', '+4'], 600)
            }
            test_df['TEST_UOFM'] = 'cells/ul'
            test_df['TEST_RSLT_num'], test_df['IS_APPRXMT'] = handle_urine_test_strips(test_df['TEST_RSLT'],
                                                                                       leu_stript_dict)

        elif test_name == 'urobilinogen ua':
            urob_dict = {
                **dict.fromkeys(['negative', 'neg', 'normal', 'norm'], 0),
                **dict.fromkeys(['trace', '<17umol/l', '<18', '<17'], 17),
                **dict.fromkeys(['small', '1+', '+1'], 35),
                **dict.fromkeys(['moderate', '2+', '+2'], 70),
                **dict.fromkeys(['large', '3+', '+3'], 140),
                **dict.fromkeys(['4+', '+4'], 200)
            }
            test_df['TEST_RSLT_num'], test_df['IS_APPRXMT'] = handle_urine_test_strips(test_df['TEST_RSLT'], urob_dict)

            # if it is not an approximate value (i.e. categorical) change the unit of those with mg/l to umol/L
            urob_molecular_weight = 592.7  # g/mol
            test_df.loc[(test_df['TEST_UOFM'] == 'mg/l') & (test_df['IS_APPRXMT'] == False), 'TEST_RSLT_num'] *= (
                    1000.0 / urob_molecular_weight)
            test_df['TEST_UOFM'] = 'umol/l'

        elif test_name == 'thyroid stimulating hormone':
            test_df['TEST_UOFM'] = 'mu/l'  # all units are the same thing
            test_df['TEST_RSLT_num'], test_df['IS_APPRXMT'] = results_to_numeric(test_df['TEST_RSLT'])

        elif test_name == 'nitrite ua':
            test_df['TEST_RSLT_num'] = np.nan
            test_df['TEST_RSLT_num'].loc[test_df['TEST_RSLT'].isin(['negative', 'neg'])] = 0
            test_df['TEST_RSLT_num'].loc[test_df['TEST_RSLT'].isin(['positive', 'pos'])] = 1
            test_df['IS_APPRXMT'] = 0

        elif test_name == 'creatinine urine random':
            test_df['TEST_RSLT_num'], test_df['IS_APPRXMT'] = results_to_numeric(test_df['TEST_RSLT'])
            test_df['TEST_RSLT_num'].loc[test_df['TEST_UOFM'] == 'umol/l'] /= 1000
            test_df['TEST_UOFM'] = 'mmol/l'

        # if the test is none of the special cases above, then use the normal procedure (only for selected tests)
        else:
            if test_name in get_lab_items():
                test_df['TEST_RSLT_num'], test_df['IS_APPRXMT'] = results_to_numeric(test_df['TEST_RSLT'])
            else:
                raise ValueError('Test name not found in lab data')

        return test_df


def get_lab_items():
    """
    This function returns the list of lab items that are used in the feature engineering process
    It uses a csv file that has the list of selected lab items
    :return: list of lab items
    """
    selected_features = pd.read_csv(cons.LAB_SELECTED_FEATURES, dtype=str)
    lab_items = selected_features['TEST_NM'].tolist()
    return lab_items


def results_to_numeric(df_res):
    """
    This function converts the results of lab tests to numeric values
    :param df_res: a series of lab results in string format
    :return: df_res_num, is_approximate. df_res_num is the numeric version of df_res. is_approximate is a boolean
    series that shows if the result is approximate or not (e.g., < 10 or test strips)
    """
    # remove < or > in the text so it removes them in case they're in front of the number
    inequal_signs = r'[<>=]\s*'
    is_approximate = df_res.str.contains(inequal_signs, regex=True)
    df_res = df_res.str.replace(inequal_signs, '', regex=True)

    # apply a regular expression to remove notes after numbers in the cells
    df_res = df_res.str.replace(r'\s*([0-9]+(\.[0-9]+)?)(?:\s+[a-zA-Z/]+.*)?$', r'\1', regex=True)
    df_res_num = pd.to_numeric(df_res, errors='coerce')
    return df_res_num, is_approximate


def handle_urine_test_strips(df_res, stript_dict):
    """
    This function converts the results of urine test strips to numeric values
    :param df_res: a series of lab results in string format
    :param stript_dict: a dictionary that maps the string results to numeric values
    Example: {'negative': 0, 'trace': 0.5, '+1': 1, 'moderate': 2, 'large': 3}
    :return: df_res_num, is_approximate. df_res_num is the numeric version of df_res. is_approximate is a boolean
    series that shows if the result is approximate or not (e.g., < 10 or test strips)
    """
    is_approximate = pd.Series([False] * len(df_res), dtype=bool)
    for key, val in stript_dict.items():
        # if it is from a test strip, then it is an approximate value
        is_approximate = is_approximate | df_res.str.contains(key, regex=False)
        df_res.loc[df_res == key] = str(val)  # first change the obvious ones
        df_res = df_res.str.replace(key, str(val), regex=False)  # then change ones with notes, etc

    df_res_num, is_approximate2 = results_to_numeric(df_res)
    is_approximate = is_approximate | is_approximate2

    return df_res_num, is_approximate
