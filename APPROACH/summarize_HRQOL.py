import APPROACH.constants as cons


def summarize_EQ5D(mobility_level, self_care_level, usual_activities_level, pain_discomfort_level,
                   anxiety_depression_level):
    """
    this function summarizes EQ5D data of a patient and calculated a summary score based on Time-Trade-Off (TTO) value.
    This function is based on the following paper for the Canadian population:
    Xie F, Pullenayegum E, Gaebel K, Bansback N, Bryan S, Ohinmaa A, Poissant L, Johnson JA;
    Canadian EQ-5D-5L Valuation Study Group. A Time Trade-off-derived Value Set of the EQ-5D-5L for Canada. Med Care.
    2016 Jan;54(1):98-105. doi: 10.1097/MLR.0000000000000447. PMID: 26492214; PMCID: PMC4674140.
    :param mobility_level: Mobility score of the patient
    :param self_care_level: Self-care score of the patient
    :param usual_activities_level: Usual activities score of the patient
    :param pain_discomfort_level: Pain/discomfort score of the patient
    :param anxiety_depression_level: Anxiety/depression score of the patient
    :return: Summary score of the patient (Health Utility based on TTO)
    """
    health_state = [mobility_level, self_care_level, usual_activities_level,
                    pain_discomfort_level, anxiety_depression_level]

    # handle missing or wrong values
    if any(level < 1 or level > 5 for level in health_state):
        return None
    if any(level is None for level in health_state):
        return None

    # Coefficients
    intercept = 1.1351
    mobility = -0.0389
    self_care = -0.0458
    usual_activities = -0.0195
    pain_discomfort = -0.0444
    anxiety_depression = -0.0376
    mobility45 = -0.0510
    self_care45 = -0.0584
    usual_activities45 = -0.1103
    pain_discomfort45 = -0.1409
    anxiety_depression45 = -0.1277
    num45sq = 0.0085

    # Main effects
    health_utility = intercept + (mobility * mobility_level) + (self_care * self_care_level) + (
                usual_activities * usual_activities_level) + (pain_discomfort * pain_discomfort_level) + (
                                 anxiety_depression * anxiety_depression_level)

    # Interaction terms for levels 4 or 5
    health_utility += mobility45 if mobility_level >= 4 else 0
    health_utility += self_care45 if self_care_level >= 4 else 0
    health_utility += usual_activities45 if usual_activities_level >= 4 else 0
    health_utility += pain_discomfort45 if pain_discomfort_level >= 4 else 0
    health_utility += anxiety_depression45 if anxiety_depression_level >= 4 else 0

    # Nonlinear term for multiple dimensions with high severity (level 4 or 5)
    num45 = sum(level >= 4 for level in health_state)
    if num45 > 1:
        health_utility += num45sq * (num45 - 1) ** 2

    return health_utility


def summarize_SAQ(items):
    """
    This function summarizes the Seattle Angina Questionnaire (SAQ) data of a patient and calculates a summary score.
    The final selected model is the summary score provided by the SAQ7 questionnaire. It includes the following items:
    - Physical Limitation (3 items): SAQ_Walking, SAQ_Gardening, SAQ_Lifting
    - Angina Frequency (2 items): SAQ43, SAQ44
    - Quality of Life (2 items): SAQ49, SAQ410
    Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4282595/
    :param items: a dictionary of the SAQ items and their values (e.g. {SAQ_Dressing': 5, 'SAQ_Walking': 3, 'SAQ_Showering': 2})
    :return: Summary score of the patient (Health Utility based on SAQ) - a number between 0 and 100
    """

    # handle missing or wrong values
    SAQ7_items = ['SAQ_Walking', 'SAQ_Gardening', 'SAQ_Lifting', 'SAQ43', 'SAQ44', 'SAQ49', 'SAQ410']
    if any(items[i] is None for i in SAQ7_items):
        return None

    physical_limitation = sum(
        100 * items[i] / cons.HRQOL_SAQ_COLS[i] for i in ['SAQ_Walking', 'SAQ_Gardening', 'SAQ_Lifting']) / 3
    angina_frequency = sum(100 * items[i] / cons.HRQOL_SAQ_COLS[i] for i in ['SAQ43', 'SAQ44']) / 2
    quality_of_life = sum(100 * items[i] / cons.HRQOL_SAQ_COLS[i] for i in ['SAQ49', 'SAQ410']) / 2

    final_score = (physical_limitation + angina_frequency + quality_of_life) / 3

    return final_score
