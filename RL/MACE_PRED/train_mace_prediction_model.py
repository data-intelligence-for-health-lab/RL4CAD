import pandas as pd
import numpy as np
import os
import training_constants as tc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import datetime

from sklearn import preprocessing
import pickle


N_EPOCHS = 1000
PATIENCE = 100
BATCH_SIZE = 128
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define the neural network architecture
class RewardEstimator(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size, activation_fn):
        super(RewardEstimator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn

        layers = [nn.Linear(input_size, hidden_size), activation_fn]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation_fn)
        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

class ModelBasedInference:
    def __init__(self, model_path, maximize_outcome=True):
        checkpoint = torch.load(model_path)
        input_size = checkpoint['parameters']['input_size']
        output_size = checkpoint['parameters']['output_size']
        num_layers = checkpoint['parameters']['num_layers']
        hidden_size = checkpoint['parameters']['hidden_size']
        activation_fn = checkpoint['parameters']['activation_fn']

        self.model = RewardEstimator(input_size, output_size, num_layers, hidden_size, activation_fn)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.maximize_outcome = maximize_outcome

    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x.astype(np.float32))
            outputs = self.model(x)
            return outputs.numpy()
        

def load_data(processed_data_path, phase):
    all_caths = pd.read_csv(os.path.join(processed_data_path, phase, 'all_caths.csv'))
    all_caths['mace'].fillna(tc.outcome_followup_time, inplace=True)
    all_caths.loc[all_caths['mace'] > tc.outcome_followup_time, 'mace'] = tc.outcome_followup_time
    # if repeat revascularization is performed, we consider it as a MACE event
    all_caths['repeated_revasc'].fillna(tc.outcome_followup_time, inplace=True)
    condition_repvasc = all_caths['repeated_revasc'] < all_caths['mace']
    all_caths.loc[condition_repvasc, 'mace'] = all_caths.loc[condition_repvasc, 'repeated_revasc']
    # if survival is <90 days, we consider it as a MACE event
    all_caths['survival'].fillna(tc.outcome_followup_time, inplace=True)
    condition_survival = (all_caths['survival'] < tc.min_acceptable_survival) & (all_caths['survival'] < all_caths['mace'])
    all_caths.loc[condition_survival, 'mace'] = all_caths.loc[condition_survival, 'survival']

    outcome = all_caths['mace'].to_numpy() / tc.outcome_followup_time
    outcome = np.clip(outcome, 0, 1)  # normalize and clip the outcome to [0, 1]

    all_caths_imputed = pd.read_csv(os.path.join(processed_data_path, phase, f'all_caths_{phase}_imputed.csv'))
    all_caths['SubsequentTreatment'].fillna('Medical Therapy', inplace=True)
    for treatment in ['CABG', 'Medical Therapy', 'PCI']:
        all_caths_imputed[treatment] = (all_caths['SubsequentTreatment'] == treatment).astype(int)

    X = torch.tensor(all_caths_imputed.values.astype(np.float32))
    y = torch.tensor(outcome.reshape(-1, 1).astype(np.float32))
    return X, y

def train_model(X_train, y_train, X_val, y_val, num_layers, hidden_size, activation_fn, epochs=N_EPOCHS, patience=PATIENCE):
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=10)

    model = RewardEstimator(X_train.shape[1], y_train.shape[1], num_layers, hidden_size, activation_fn)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_val_loss += criterion(outputs, labels).item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {avg_val_loss}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    return model

def evaluate_model(model, X_val, y_val):
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=10)

    criterion = nn.MSELoss()
    total_val_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_val_loss += criterion(outputs, labels).item()

    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss

def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # Define the hyperparameter space
    num_layers_options = [2, 3, 4]
    hidden_size_options = [8, 16, 32, 64, 128]
    activation_functions = [nn.ReLU(), nn.LeakyReLU()]

    best_val_loss = float('inf')
    best_hyperparams = {}

    for num_layers in num_layers_options:
        for hidden_size in hidden_size_options:
            for activation_fn in activation_functions:
                print(f'Training model with {num_layers} layers, {hidden_size} hidden units, and {activation_fn}')
                model = train_model(X_train, y_train, X_val, y_val, num_layers, hidden_size, activation_fn)
                val_loss = evaluate_model(model, X_val, y_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_hyperparams = {
                        'num_layers': num_layers,
                        'hidden_size': hidden_size,
                        'activation_fn': activation_fn
                    }

    print(f'Best Hyperparameters: {best_hyperparams}')
    return best_hyperparams

# Main function to execute the training and tuning
def train_mace_prediction_model():
    processed_data_path = tc.processed_data_path
    reward_name = 'mace-survival-repvasc'

    X_train, y_train = load_data(processed_data_path, 'train')
    X_val, y_val = load_data(processed_data_path, 'validation')
    X_test, y_test = load_data(processed_data_path, 'test')

    best_hyperparams = hyperparameter_tuning(X_train, y_train, X_val, y_val)
    best_model = train_model(X_train, y_train, X_val, y_val, **best_hyperparams)

    # Evaluate the model on the test set
    test_loss = evaluate_model(best_model, X_test, y_test)
    print(f'Test Loss: {test_loss}')

    # Save the model
    models_path = os.path.join(tc.models_path, f'{reward_name}_prediction_model')
    os.makedirs(models_path, exist_ok=True)
    model_path = os.path.join(models_path, f'model_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.pth')
    checkpoint = {
        'state_dict': best_model.state_dict(),
        'parameters': {'input_size': X_train.shape[1],
                        'output_size': y_train.shape[1],
                        **best_hyperparams},
        'test_loss': test_loss}
    torch.save(checkpoint, model_path)
    print(f'Model saved at {model_path}')

if __name__ == '__main__':
    train_mace_prediction_model()

