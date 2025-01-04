# RLTL
## Addressing Distribution Shifts in Optimization of Coronary Artery Disease Treatment Using Reinforcement and Transfer Learning


This repository contains the codes and resources related to the following paper:

    @article{ghasemi2024RLTL,
    title={Addressing Distribution Shifts in Optimization of Coronary Artery Disease Treatment Using Reinforcement and Transfer Learning},
    author={Ghasemi, Peyman and Greenberg, Matthew and Southern, Danielle A and Li, Bing and White, James A and Lee, Joon},
    year={2024},
    journal={Preprint}
    }


### Abstract
```
# Importance
Optimizing revascularization strategies for patients with obstructive coronary artery disease (CAD) is crucial for improving outcomes. Yet, personalized treatment is hindered by variations in clinical practice and patient characteristics.

# Objective
To determine whether reinforcement learning (RL) models can optimize coronary revascularization decisions across different patient subgroups and settings and assess if transfer learning (TL) can mitigate distribution shifts.
 
# Design
Retrospective cohort study using data from the Alberta Provincial Project for Outcome Assessment in Coronary Heart Disease (APPROACH) Registry from 2009 to 2019. Offline RL modeling was used for optimization and off-policy evaluation (OPE) assessed physician behavior and optimized RL policies across stratified groups. TL techniques addressed distribution shifts.
 
# Setting
Three cardiac catheterization sites in Alberta, Canada.
 
# Participants
A total of 41,328 adults with obstructive CAD who underwent 49,521 cardiac catheterizations between 2009 and 2019. Patients with ST-elevation myocardial infarction were excluded. Data were stratified by sex and by site, excluding patients appearing in multiple sites under site-based stratification.
 
# Main Outcomes and Measures
The primary outcome was the expected reward, representing treatment outcomes measured by major adverse cardiovascular events, estimated using OPE of different policies. The study evaluated the impact of distribution shifts on physician behavior and optimized RL policies across different patient groups and assessed the effectiveness of TL in mitigating these shifts.
 
# Results
Female patients had significantly lower expected rewards under their own physician behavior policies compared to males. Applying the male behavior policy improved rewards for females but remained lower than males' outcomes. Optimized RL policies outperformed physician behavior policies but were affected by distribution shifts when applied across groups. TL, fine-tuning RL models with as little as 1% of target group data, effectively mitigated distribution shifts and achieved outcomes close to models trained on full target data.
 
# Conclusions and Relevance
RL modeling optimized coronary revascularization strategies, outperforming physician behavior policies despite practice variations. TL efficiently adapted RL models to new patient groups using minimal data, demonstrating potential to reduce disparities and improve personalized CAD treatment. These findings suggest that RL and TL can advance individualized care and promote equitable healthcare delivery.
```


### Files and Folders Structure
```
.
├── README.md
├── evaluate_on_opposing_CQL_policy.py
├── evaluate_on_opposing_behavior_policy.py
├── find_behavior_policy.py 
├── process_data.py 
├── train_cql.py    
├── transfer_constants.py   # Constants required for the code
├── transfer_learning_on_cql.py
└── transfer_learning_on_cql_frozen_encoder.py
```

### Steps to run
- Make sure RL4CAD and APPRAOCH codebases are functioning correctly and install the requirements
- Stratify the patients and save the lists of patient identifier numbers in CSV files. Refer to the `transfer_constants.py` for details
- Run these codes:
    - `process_data.py`: Initial data processing (aggregation of data for RL) for stratified groups
    - `find_behavior_policy.py`: Estimate the behavior policy for each group
    - `evaluate_on_opposing_behavior_policy.py`: Experiment 1 - WIS of each behavior policy on different groups
    - `train_cql.py`: # Train CQL models on each group (with hyperparameter tuning) and save the models
    - `evaluate_on_opposing_CQL_policy.py`: Experiment 2 - WIS of each CQL policy on different groups
    - Either of these two:
        -  `transfer_learning_on_cql_frozen_encoder.py`: Uses the chosen CQL policy for the largest group to do Transfer learning (experiment 3)
        - `transfer_learning_on_cql.py`: Same, but no encoder freezing - Not included in the paper.

Each code includes an argument `--stratify_on` which can be either `hospital`, `sex`, or whatever you have stratified and included the required info in `transfer_constants.py`.


### Contact
For any inquiries regarding the code, please contact the DIH Lab or the author via [LinkedIn](https://www.linkedin.com/in/pghasemi/). You can also open an issue in this repository.
