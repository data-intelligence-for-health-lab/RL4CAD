# ICD codes
import simple_icd_10 as icd


def is_cardiovascular_death_icd10(icd10_code: str):
    """
    This method gets an ICD10 code for the cause of death and determines whether it is Cardiovascular or not.
    :param icd10_code: String form of an ICD10 code
    :return: True for cardiovascular death, otherwise False
    """
    if not isinstance(icd10_code, str):
        icd10_code = str(icd10_code)
    if icd10_code == 'nan':
        return False

    cardiovascular_death_icd10_list = ['I05', 'I06', 'I07', 'I08', 'I09', 'I2', 'I3', 'I4', 'I5', 'I6']
    if any(icd10_code.lower().startswith(icd.lower()) for icd in cardiovascular_death_icd10_list):
        return True
    else:
        return False


def is_acute_myocardial_infarction(icd10_code: str):
    """
    This method gets an ICD10 code and determines if it is related to an acute myocardial infarction or not
    acute myocardial infarction and subsequent myocardial infarction are included as positive.
    Source: https://icd.who.int/browse10/2019/en#/I20-I25

    :param icd10_code: String form of an ICD10 code
    :return: True for AMI, otherwise False
    """
    if not isinstance(icd10_code, str):
        icd10_code = str(icd10_code)
    if icd10_code == 'nan':
        return False

    acute_myocardial_infarction_list = ['I21', 'I22']
    if any(icd10_code.lower().startswith(icd.lower()) for icd in acute_myocardial_infarction_list):
        return True
    else:
        return False


def is_stroke(icd10_code: str):
    """
    This method gets an ICD10 code and determines if it is related to a stroke or not
    :param icd10_code: String form of an ICD10 code
    :return: True for stroke, otherwise False
    """
    if not isinstance(icd10_code, str):
        icd10_code = str(icd10_code)
    if icd10_code == 'nan':
        return False

    stroke_list = ['I63']
    if any(icd10_code.lower().startswith(icd.lower()) for icd in stroke_list):
        return True
    else:
        return False


def is_icd_code_equal_to(icd10_code: str, ref: str):
    """
    This method gets an ICD10 code and determines if it is equal to a reference code or not
    To do this, it checks if the code is equal to the reference code or if the reference code is an ancestor of the code
    :param icd10_code: String form of an ICD10 code
    :param ref: String form of a reference code
    :return: True if equal, otherwise False
    """
    ref = ref.replace('.', '')  # remove dots from the reference code
    if not isinstance(icd10_code, str):
        icd10_code = str(icd10_code)
    if icd10_code == 'nan':
        return False

    is_equal = False
    # If the code is not valid (maybe it is Canadian), we remove the last character and check again
    while not icd.is_valid_item(icd10_code):
        if len(icd10_code) == 0:
            return False
        if icd10_code == ref:
            return True
        icd10_code = icd10_code[:-1]

    # if the code is valid, we check if it is equal to the reference code
    # or if the reference code is an ancestor of the code
    icd10_code = icd.remove_dot(icd10_code)

    if icd10_code == ref:
        is_equal = True
    else:
        if icd.is_valid_item(icd10_code):
            ancestors = icd.get_ancestors(icd10_code)
            ancestors = [icd.remove_dot(ancestor) for ancestor in ancestors]
            if ref in ancestors:
                is_equal = True

    return is_equal
