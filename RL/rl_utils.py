
import numpy as np
import os
import torch
import d3rlpy
import training_constants as tc
import pandas as pd
import datetime
from scipy import stats
from train_mace_prediction_model import ModelBasedInference



class Transition:
    def __init__(self, state, action, reward, done, next_state=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.next_state = next_state

        # the prediction probabilities for the action taken in the state.
        # This is actually the RL model prediction for each action in the state
        # It is used for importance sampling
        # it must be a dictionary of policies with a list of probabilities for each action
        # (e.g., {'policy1': [0.1, 0.2, 0.7], 'policy2': [0.3, 0.3, 0.4]})
        self.prediction_probs = {}  

    def __str__(self):
        return "state: {}, action: {}, reward: {}, done: {}".format(
            self.state, self.action, self.reward, self.done
        )
    
class Episode:
    def __init__(self, patient_id=None):
        self.transitions = []
        if patient_id is not None:
            self.patient_id = patient_id
        else:
            self.patient_id = None

    def add_transition(self, transition):
        self.transitions.append(transition)

    def __str__(self):
        return "Episode: {}".format(self.transitions.__str__())
    
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        return self.transitions[idx]
    
    def __iter__(self):
        return iter(self.transitions)
    

class ClusteringBasedInference:
    def __init__(self, model, n_states, policy_probs) -> None:
        """
        A class to perform inference for a clustering-based MDP model
        Args:
        model: the clustering model (e.g., KMeans)
        n_states: the number of states (clusters)
        policy: the optimal policy for the MDP
        """
        self.model = model
        self.n_states = n_states
        self.optimal_policy = policy_probs

    def get_state(self, X):
        """
        Get the state of the input data
        Args:
        X: the input data
        """
        return self.model.predict(X)
    
    def get_policy(self, X):
        """
        Get the policy for the input data
        Args:
        X: the input data
        """
        state = self.get_state(X)
        return self.optimal_policy[state,:]
    

def weighted_importance_sampling_with_bootstrap(episodes, gamma, policy, behavior_policy, num_bootstrap_samples, N, confidedence_level=95, reward_name=None):
    """
    Computes weighted importance sampling estimate with bootstrap confidence intervals
    idea from https://github.com/IntelLabs/coach/blob/master/rl_coach/off_policy_evaluators/rl/weighted_importance_sampling.py

    Args:
    :param episodes: list of episodes
    :param gamma: discount factor
    :param policy: target policy
    :param behavior_policy: behavior policy
    :param num_bootstrap_samples: number of bootstrap samples to use
    :param N: number of episodes to sample for each bootstrap sample
    :param confidence_level: confidence level for confidence intervals
    :param reward_name: name of reward to use (if None, reward is assumed to be a number, and not a dictionary)

    Returns:
    :return: weighted importance sampling estimate, bootstrap confidence interval
    """
    # Precompute weights and rewards for all episodes
    episode_weights, episode_rewards = precompute_weights_rewards(episodes, gamma, policy, behavior_policy, reward_name)

    bootstrap_wis_estimates = []
    for _ in range(num_bootstrap_samples):
        sampled_indices = np.random.choice(len(episodes), N, replace=True)
        wis_estimate = compute_wis_for_sample(sampled_indices, episode_weights, episode_rewards)
        bootstrap_wis_estimates.append(wis_estimate)

    # lower_bound = np.percentile(bootstrap_wis_estimates, (100 - confidedence_level) / 2)
    # upper_bound = np.percentile(bootstrap_wis_estimates, 100 - (100 - confidedence_level) / 2)

    wis_mean = np.mean(bootstrap_wis_estimates)
    wis_sem = stats.sem(bootstrap_wis_estimates)
    conf_interval = stats.norm.interval(confidedence_level / 100, loc=wis_mean, scale=wis_sem)
    lower_bound, upper_bound = conf_interval

    return np.mean(bootstrap_wis_estimates), (lower_bound, upper_bound)

def precompute_weights_rewards(episodes, gamma, policy, behavior_policy, reward_name):
    """
    Precomputes weights and rewards for all episodes

    Args:
    :param episodes: list of episodes
    :param gamma: discount factor
    :param policy: target policy
    :param behavior_policy: behavior policy
    :param reward_name: name of reward to use

    Returns:
    :return: episode weights and episode rewards
    """
    epsilon = 1e-10
    episode_weights = []
    episode_rewards = []
    for episode in episodes:
        w_i = 1
        total_reward = 0
        for j, transition in enumerate(episode.transitions):
            a = transition.action
            # w_i *= policy[s, a] / behavior_policy[s, a]
            # w_i *= policy.predict_prob(s, a) / (behavior_policy.predict_prob(s, a) + epsilon)
            w_i *= policy.predict_prob(transition, a) / (behavior_policy.predict_prob(transition, a) + epsilon)
            if reward_name is not None:
                total_reward += transition.reward[reward_name] * gamma ** j
            else:
                total_reward += transition.reward * gamma ** j

        episode_weights.append(w_i)
        episode_rewards.append(total_reward)
    return episode_weights, episode_rewards

def compute_wis_for_sample(sample_indices, episode_weights, episode_rewards):
    """
    Computes weighted importance sampling estimate for a sample of episodes
    
    Args:
    :param sample_indices: indices of episodes to sample
    :param episode_weights: list of weights for each episode
    :param episode_rewards: list of rewards for each episode

    Returns:
    :return: weighted importance sampling estimate
    """
    sampled_weights = [episode_weights[i] for i in sample_indices]
    sampled_rewards = [episode_rewards[i] for i in sample_indices]

    total_weight = np.sum(sampled_weights)
    if total_weight == 0:
        return 0

    wis = np.sum(np.array(sampled_weights) * np.array(sampled_rewards)) / total_weight
    return wis


def softmax(Q_matrix):
    """
    Computes softmax over Q values

    Args:
    :param Q_matrix: matrix of Q values with shape (num_states, num_actions)

    Returns:
    :return: softmax over Q values
    """
    e_Q = np.exp(Q_matrix - np.max(Q_matrix, axis=1, keepdims=True)) # Subtracting max for numerical stability
    return e_Q / e_Q.sum(axis=1, keepdims=True)


class PolicyResolver:
    def __init__(self, model, action_space, greedy=False):
        if isinstance(model, dict):
            self.model_type = dict
        elif isinstance(model, np.ndarray):
            self.model_type = np.ndarray
        elif isinstance(model, np.matrix):
            self.model_type = np.matrix

        elif isinstance(model, torch.nn.Module):
            self.model_type = torch.nn.Module
            model.eval()
        
        elif isinstance(model, ModelBasedInference):
            self.model_type = ModelBasedInference

        elif isinstance(model, ClusteringBasedInference):
            self.model_type = ClusteringBasedInference

        elif isinstance(model, str):
            self.model_type = str


        # elif isinstance(model, d3rlpy.algos.dqn.DQN):
        #     self.model_type = d3rlpy.algos.dqn.DQN
        elif "d3rlpy.algos" in model.__class__.__module__:
            self.model_type = "d3rlpy.algos"

        self.model = model
        self.action_space = action_space
        self.greedy = greedy

    def predict_prob(self, transition, action):
        state = transition.state
        if self.model_type == dict:
            probs = [self.model[state][a] for a in self.action_space]
            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1
            return probs[action]
        
        elif self.model_type == str:
            probs = transition.prediction_probs[self.model]
            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1
            return probs[action]
        
        elif self.model_type == np.ndarray:
            # if state in an a list or array, then just use the first element
            if isinstance(state, (list, np.ndarray)):
                state = int(state[0])
            probs = self.model[state, :]
            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1
            return probs[action]
        
        # elif self.model_type == np.matrix:
        #     probs = self.model[state, :]
        #     if self.greedy:
        #         probs = np.zeros(len(self.action_space))
        #         probs[np.argmax(probs)] = 1
        #     return probs[0, action]
        
        elif self.model_type == torch.nn.Module:
            with torch.no_grad():
                sample = torch.tensor(state.astype(np.float32).reshape(1, -1))
                logits = self.model(sample)
                probs = torch.nn.functional.softmax(logits, dim=1).numpy()
                probs = probs.reshape(-1)

            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1

            return probs[action]
            
        elif self.model_type == ModelBasedInference:
            samples = []
            for i in range(len(self.action_space)):
                action_dummies = np.zeros(len(self.action_space))
                action_dummies[i] = 1
                state_action = np.concatenate([state, action_dummies])
                samples.append(state_action)
            samples = np.array(samples)
            rewards = self.model.predict(samples)
            if not self.model.maximize_outcome:  # if the model is trained to minimize the outcome, then reverse the rewards
                rewards = -rewards
            rewards = rewards.reshape(-1)
            probs = torch.nn.functional.softmax(torch.tensor(rewards), dim=0).numpy()

            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1

            return probs[action]
        
        elif self.model_type == ClusteringBasedInference:
            # get the probs from the model
            probs = self.model.get_policy(state.reshape(1, -1))

            if self.greedy:
                probs = np.zeros(len(self.action_space))
                probs[np.argmax(probs)] = 1

            return probs[action]
        
        # elif self.model_type == d3rlpy.algos.dqn.DQN:
        elif self.model_type == "d3rlpy.algos":
            observation = np.tile(state.astype(np.float32), (len(self.action_space), 1))
            logits = self.model.predict_value(observation, np.array(self.action_space))
            # normalize the logits to their summation to get the probabilities
            # probs  = logits / np.sum(logits)
            
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy()

            if self.greedy:
                max_action = np.argmax(probs)
                probs = np.zeros(len(self.action_space))
                probs[max_action] = 1

            return probs[action]


# def reward_func_mace(reward_dic, followup_time):
#     survival = reward_dic['survival']
#     mace = reward_dic['mace']
#     if survival > followup_time:
#         survival = followup_time
#     if mace > followup_time:
#         mace = followup_time
    
#     if survival < 90:
#         reward = -1
#     else:
#         reward = (survival / followup_time) - (1 - mace / followup_time)
    
#     return reward

def reward_func_mace(reward_dic, followup_time, is_terminal_state=False, **kwargs):
    """
    Reward function for MACE 
    """
    survival = reward_dic['survival']
    mace = reward_dic['mace']
    if survival > followup_time:
        survival = followup_time
    if mace > followup_time:
        mace = followup_time
    
    if not is_terminal_state:
        reward = -(1 - mace / followup_time)
    else:
        reward = (survival / followup_time) - (1 - mace / followup_time)

    return reward

def reward_func(reward_dic, followup_time, is_terminal_state=False, action=None, cost_adjustment_coef=1.0, **kwargs):
    """
    Reward function for MACE adjusted for cost
    """
    action_cost_ratios = tc.cost_ratios  # a dictionary of costs for each action
    cost = action_cost_ratios[action]

    survival = reward_dic['survival']
    mace = reward_dic['mace']
    if survival > followup_time:
        survival = followup_time
    if mace > followup_time:
        mace = followup_time
    
    if not is_terminal_state:
        reward = -(1 - mace / followup_time) - cost * cost_adjustment_coef
    else:
        reward = (survival / followup_time) - (1 - mace / followup_time) - cost * cost_adjustment_coef

    return reward

def reward_func_mace_survival_cost(reward_dic, followup_time, is_terminal_state=False, action=None, cost_adjustment_coef=1.0, **kwargs):
    """
    Reward function for MACE and survival adjusted for cost
    """
    action_cost_ratios = tc.cost_ratios  # a dictionary of costs for each action
    cost = action_cost_ratios[action]

    survival = reward_dic['survival']
    mace = reward_dic['mace']
    if survival > followup_time:
        survival = followup_time
    if mace > followup_time:
        mace = followup_time

    # consider 90-day mortality as a MACE event (if the mace event has not already happened)
    if (survival < tc.min_acceptable_survival) and (mace > tc.min_acceptable_survival):
        mace = survival
    
    if not is_terminal_state:
        reward = -(1 - mace / followup_time) - cost * cost_adjustment_coef
    else:
        reward = (survival / followup_time) - (1 - mace / followup_time) - cost * cost_adjustment_coef

    return reward

def reward_func_mace_survival(reward_dic, followup_time, is_terminal_state=False, action=None, **kwargs):
    """
    Reward function for MACE and survival (with no cost adjustment)
    """
    cost_adjustment_coef = 0.0
    reward = reward_func_mace_survival_cost(reward_dic, followup_time, is_terminal_state, action, cost_adjustment_coef, **kwargs)

    return reward

def reward_func_mace_survival_repvasc_cost(reward_dic, followup_time, is_terminal_state=False, action=None, cost_adjustment_coef=1.0, **kwargs):
    """
    Reward function for MACE and survival adjusted for cost
    """
    action_cost_ratios = tc.cost_ratios  # a dictionary of costs for each action
    cost = action_cost_ratios[action]

    survival = reward_dic['survival']
    repvasc = reward_dic['repeated_revasc']
    mace = reward_dic['mace']
    if survival > followup_time:
        survival = followup_time
    if mace > followup_time:
        mace = followup_time
    if repvasc > followup_time:
        repvasc = followup_time

    # if repeated revascularization is smaller than the mace event, then consider it as the mace event
    if repvasc < mace:
        mace = repvasc

    # consider 90-day mortality as a MACE event (if the mace event has not already happened)
    if (survival < tc.min_acceptable_survival) and (mace > tc.min_acceptable_survival):
        mace = survival
    
    if not is_terminal_state:
        reward = -(1 - mace / followup_time) - cost * cost_adjustment_coef
    else:
        reward = (survival / followup_time) - (1 - mace / followup_time) - cost * cost_adjustment_coef

    return reward

def reward_func_mace_survival_repvasc(reward_dic, followup_time, is_terminal_state=False, action=None, **kwargs):
    """
    Reward function for MACE and survival (with no cost adjustment)
    """
    cost_adjustment_coef = 0.0
    reward = reward_func_mace_survival_repvasc_cost(reward_dic, followup_time, is_terminal_state, action, cost_adjustment_coef, **kwargs)

    return reward


def get_episodes(all_caths_, all_caths_imputed, action_encoder, rewards_list, reward_func):
    """
    Create episodes from the all_caths data. It is based on the Episode class and the Transition class defined in this module.
    
    """
    all_caths = all_caths_.copy()

    # encode the actions
    all_caths['SubsequentTreatmentCode'] = action_encoder.transform(all_caths['SubsequentTreatment'].values)

    # replace the features with the imputed features
    features = all_caths_imputed.columns
    all_caths[features] = all_caths_imputed[features]

    # sort the data by patient and procedure number
    all_caths.sort_values(by=['File No', 'Procedure Number'], ignore_index=True, inplace=True)

    # create the episodes
    episodes = []

    for file_no, group in all_caths.groupby('File No'):
        start_new_episode_flag = True  # flag to indicate if the next procedure is the start of a new episode
        for idx, row in group.iterrows():
            if start_new_episode_flag:
                episode = Episode(patient_id=file_no)
                start_new_episode_flag = False

            state = row[features].to_numpy()
            action = row['SubsequentTreatmentCode']
            done = False
            # if this is the last procedure for the file_no, then set done to True
            if idx == group.index[-1]:
                done = True
                next_state = None
                start_new_episode_flag = True
            else:
                current_procedure_date = pd.to_datetime(row['Procedure Standard Time'])
                next_procedure_date = pd.to_datetime(group.loc[idx + 1, 'Procedure Standard Time'])
                next_state = group.loc[idx + 1, features].to_numpy()
                # if the time between the current and next procedure is greater than the outcome_followup_time, then set done to True and start a new episode
                if next_procedure_date - current_procedure_date > pd.Timedelta(tc.outcome_followup_time, unit='day'):
                    done = True
                    start_new_episode_flag = True

            reward_dic = {r: row[r] for r in rewards_list}
            reward = reward_func(reward_dic, followup_time=tc.outcome_followup_time, is_terminal_state=done, action=row['SubsequentTreatment'])
            
            transition = Transition(state, action, reward, done, next_state)
            episode.add_transition(transition)
            if start_new_episode_flag:
                episodes.append(episode)

    return episodes


def get_d3rlpy_dataset(episodes, actions_dict):
    """
    Create a d3rlpy dataset from the episodes
    """
    
    observations = []
    actions = []
    rewards = []
    terminals = []
    for episode in episodes:
        for transition in episode:
            observations.append(transition.state)
            actions.append(transition.action)
            rewards.append(transition.reward)
            terminals.append(transition.done)

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    terminals = np.array(terminals)

    dataset = d3rlpy.dataset.MDPDataset(observations, actions, rewards, terminals, action_space=d3rlpy.ActionSpace.DISCRETE, action_size=len(actions_dict))

    return dataset
                
            