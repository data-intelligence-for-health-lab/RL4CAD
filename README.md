# RL4CAD
## RL4CAD: Personalized Decision Making for Coronary Artery Disease Treatment using Offline Reinforcement Learning

This repository contains the codes and resources related to the following paper:

    @article{ghasemi2024RL4CAD,
    title={Personalized Decision Making for Coronary Artery Disease Treatment using Offline Reinforcement Learning},
    author={Ghasemi, Peyman and Greenberg, Matthew and Southern, Danielle A and Li, Bing and White, James A and Lee, Joon},
    year={2024},
    journal={Preprint}
    }


### Abstract
Choosing optimal revascularization strategies for patients with obstructive coronary artery disease (CAD) remains a clinical challenge. While randomized controlled trials offer population-level insights, gaps remain regarding personalized decision-making for individual patients. We applied off-policy reinforcement learning (RL) to a composite data model from 41,328 unique patients with angiography-confirmed obstructive CAD. In an offline setting, we estimated optimal treatment policies and evaluated these policies using weighted importance sampling. Our findings indicate that RL-guided therapy decisions outperformed physician-based decision making, with RL policies achieving up to 32% improvement in expected rewards based on composite major cardiovascular events outcomes. Additionally, we introduced methods to ensure that RL CAD treatment policies remain compatible with locally achievable clinical practice models, presenting an interpretable RL policy with a limited number of states. Overall, this novel RL-based clinical decision support tool, RL4CAD, demonstrates potential to optimize care in patients with obstructive CAD referred for invasive coronary angiography.

### Model Weights
https://drive.google.com/drive/folders/1SuDCfoNeZWBifAFqBurDAbgfR7Xm2m5x?usp=sharing


### Requirements
1- `pip install -r requirements.txt`

2- Propagate `RL/patiens_list.csv` with patient identifier numbers and flags indicating whether they are processed.

### Files and Folders Structure
```
.
├── APPROACH    # Codebase to clean, analyze, and curate APPROACH (Alberta Provincial Project for Outcome Assessment in Coronary Heart disease) and connected AHS Administrative datasets
│   ├── APPROACH.py # APPROACH dataset and Patient class definitions
│   ├── ICDcodeHelper.py    # Helper functions to analyze ICD codes
│   ├── PandasCadExtension.py   # Pandas extension class specific for this project
│   ├── approach_cleaning_helper.py # Helper functions to clean main approach dataset
│   ├── constants.py    # Constants within the project
│   ├── lab_data_cleaning_helper.py # Helper functions to clean LAB dataset (AHS Admin)
│   ├── selected_features   # Folder including selected features for AHS Admin datasets (for DAD, NACRS, PIN, refer to our previous study cited in the paper)
│   │   ├── DAD
│   │   │   └── concrete_with_weights.csv
│   │   ├── LAB
│   │   │   ├── number_of_patients_per_test.csv
│   │   │   └── number_of_patients_per_test_old.csv
│   │   ├── NACRS
│   │   │   └── concrete_with_weights.csv
│   │   └── PIN
│   │       └── concrete_with_weights.csv
│   └── summarize_HRQOL.py  # Helper functions for Health-related Quality of Life data (N/A for this paper)
├── LICENSE
├── README.md
├── RL  # Main Codebase for RL4CAD study
│   ├── DQN_CQL # train and evaluate DQN and CQL models
│   │   └── dqn.py
│   ├── MACE_PRED   # train and evaluate a simple policy based on a MACE prediction neural-network.
│   │   ├── evaluation_on_prediction_model.py
│   │   └── train_mace_prediction_model.py
│   ├── QL  # train and evaluate traditional QL models
│   │   ├── autoencoder_kmeans_RL.py
│   │   ├── autoencoder_sigmoid_training.py
│   │   └── shap_on_states.py
│   ├── data_prep_pipeline.py
│   ├── do_kmeans.py
│   ├── find_best_behavior_policy.py
│   ├── pull_and_aggregate_data_for_each_patient.py
│   ├── rl_utils.py
│   └── training_constants.py
└── RLTL    # Main Codebase for RLTL study (another study, though depending on RL4CAD) / Explained in its own README file
    ├── README.md
    ├── evaluate_on_opposing_CQL_policy.py
    ├── evaluate_on_opposing_behavior_policy.py
    ├── find_behavior_policy.py
    ├── process_data.py
    ├── retrain_CQL_on_best_hyp.py
    ├── stratify_patients.py
    ├── train_cql.py
    ├── transfer_constants.py
    ├── transfer_learning_on_cql.py
    └── transfer_learning_on_cql_frozen_encoder.py
```

### Important Files
 
#### `pull_and_aggregate_data_for_each_patient`:
This script connects to a larger class in the `APPROACH` folder and pulls and aggregates data for each patient from various databases in my cohort. It’s cohort-specific, so you may not need it. It saves each patient’s data in a separate folder.

#### `data_prep_pipeline`:
Prepares the data as Train/Validation/Test sets in a single file. It creates two files: one with raw features and other metadata, and another with cleaned and imputed features ready for ML.

#### `do_kmeans`:
Performs k-means clustering with different numbers of clusters (K) on the training set and saves the cluster assignments for each sample (train or test) in a file for easier access. This function is already called from the data_prep_pipeline.

#### `find_best_behavior_policy`:
Based on the clustering results, this script finds the optimal K that maximizes the accuracy of predicting the behavior action. You can use this best number of clusters later as your behavior policy.

#### `QL/autoencoder_kmeans_RL`:
Implements traditional Q-learning on different clustering results (where each cluster represents a state). It also has an option to use an autoencoder at the beginning, though I didn’t use it in my paper. The code is optimized for multi-core processing for efficiency, which sacrificed the readability.

#### `DQN_CQL/dqn.py`:
Implements both DQN and CQL (selectable via input) and evaluates the models. It is based on d3rlpy library. It performs a grid search on your chosen hyperparameters and saves the results in a CSV file. You can use it to select the best hyperparameter set.

#### `rl_utils`:
This might be the most useful file for you. It has the class definitions for Transitions, Episodes, implementations of Weighted Importance Sampling, reward functions, forming episodes from raw data, and more.


### Contact
For any inquiries regarding the code, please contact the DIH Lab or the author via [LinkedIn](https://www.linkedin.com/in/pghasemi/). You can also open an issue in this repository.