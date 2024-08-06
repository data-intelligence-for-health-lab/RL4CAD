import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import training_constants as tc
import logging
import datetime
from torch.utils.data import DataLoader, TensorDataset

torch.cuda.set_device(3)
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(filename='autoencoder_training_logs.log',  # Set the log file name
                        level=logging.DEBUG,     # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                        format='%(asctime)s [%(levelname)s]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

logging.info(f'Start the program at {datetime.datetime.now()}')



# Load the data
BATCH_SIZE = 128


processed_data_path = tc.processed_data_path

all_caths_train_imputed = pd.read_csv(os.path.join(processed_data_path, 'train', 'all_caths_train_imputed.csv'))

all_caths_validation_imputed = pd.read_csv(os.path.join(processed_data_path, 'validation', 'all_caths_validation_imputed.csv'))

all_caths_test_imputed = pd.read_csv(os.path.join(processed_data_path, 'test', 'all_caths_test_imputed.csv'))

# Assuming all_caths_train, all_caths_validation, and all_caths_test are already loaded and preprocessed
X_train = torch.tensor(all_caths_train_imputed.values).float()
X_val = torch.tensor(all_caths_validation_imputed.values).float()
X_test = torch.tensor(all_caths_test_imputed.values).float()

# Convert datasets to PyTorch DataLoader for batch processing
train_dataset = TensorDataset(X_train, X_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(X_val, X_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

logging.info(f'X_train shape: {X_train.shape}')
logging.info(f'X_val shape: {X_val.shape}')
logging.info(f'X_test shape: {X_test.shape}')

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_hidden_layers, activation_fn):
        super(Autoencoder, self).__init__()
        # Encoder
        encoder_layers = [nn.Linear(input_dim, latent_dim * 2 ** num_hidden_layers)]
        encoder_layers += [activation_fn()]
        for i in range(num_hidden_layers, 0, -1):
            encoder_layers.append(nn.Linear(latent_dim * 2 ** i, latent_dim * 2 ** (i - 1)))
            # the latent layer will have a sigmoid activation function (to ensure the output is between 0 and 1)
            # and the rest will have ReLU/LeakyReLU activation functions
            if i == 1:
                encoder_layers.append(nn.Sigmoid())
            else:
                encoder_layers.append(activation_fn())

        # Decoder
        decoder_layers = []
        for i in range(num_hidden_layers):
            decoder_layers.append(nn.Linear(latent_dim * 2 ** i, latent_dim * 2 ** (i + 1)))
            decoder_layers.append(activation_fn())
        decoder_layers.append(nn.Linear(latent_dim * 2 ** num_hidden_layers, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

def train_autoencoder(train_loader, val_loader, latent_dim, num_hidden_layers, activation_fn, best_loss, num_epochs=3000, patience=10, min_epochs=100):
    model = Autoencoder(X_train.shape[1], latent_dim, num_hidden_layers, activation_fn).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # best_epoch_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, X_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                val_output = model(X_batch)
                val_loss += criterion(val_output, X_batch).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch}, Val Loss: {val_loss}")

        if val_loss < best_loss:
            best_loss = val_loss
            autoencoder_path = os.path.join(tc.models_path, f'autoencoder_sigmoid_{latent_dim}')
            os.makedirs(autoencoder_path, exist_ok=True)
            checkpoint = {
                'state_dict': model.state_dict(),
                'parameters': {'input_dim': X_train.shape[1],
                                'latent_dim': latent_dim,
                                'num_hidden_layers': num_hidden_layers,
                                'activation_fn': activation_fn},
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }
            torch.save(checkpoint, os.path.join(autoencoder_path, 'checkpoint.pth'))
            # torch.save(model.state_dict(), os.path.join(autoencoder_path, 'autoencoder.pth'))
            # torch.save(model.encoder.state_dict(), os.path.join(autoencoder_path, 'encoder.pth'))
            logging.info(f"Model saved at epoch {epoch} for latent dimension {latent_dim}")
            patience_counter = 0
        else:
            if epoch >= min_epochs:  # Early stopping only after a minimum number of epochs
                patience_counter += 1
                if patience_counter > patience:
                    print(f"Stopping early at epoch {epoch} for latent dimension {latent_dim}")
                    logging.info(f"Stopping early at epoch {epoch} for latent dimension {latent_dim}")
                    break

    return best_loss

if __name__ == '__main__':
    best_overall_loss = float('inf')
    best_latent_dim = None

    for latent_dim in [64]: #[8, 16, 32, 64, 128]:
        best_loss_for_latent_dim = float('inf')
        for num_hidden_layers in [0, 1, 2, 3, 4]:
            for activation_fn in [nn.ReLU, nn.LeakyReLU]:
                loss = train_autoencoder(train_loader, val_loader, latent_dim, num_hidden_layers, activation_fn, best_loss_for_latent_dim)
                logging.info(f"Latent dim: {latent_dim}, Num layers: {num_hidden_layers}, Activation: {activation_fn}, Loss: {loss}")
                if loss < best_overall_loss:
                    best_overall_loss = loss
                    best_latent_dim = latent_dim
        logging.info(f"Best loss so far: {best_overall_loss} for latent dimension {best_latent_dim}. Time: {datetime.datetime.now()}")

    # Loading the best model overall
    best_model = Autoencoder(X_train.shape[1], best_latent_dim, num_hidden_layers, activation_fn)
    best_model.load_state_dict(torch.load(os.path.join(tc.models_path, f'autoencoder_{best_latent_dim}', 'checkpoint.pth'))['state_dict'])
    encoder_model = best_model.encoder
    best_model_path = os.path.join(tc.models_path, 'best_encoder')
    os.makedirs(best_model_path, exist_ok=True)
    torch.save(encoder_model.state_dict(), os.path.join(best_model_path, 'encoder.pth'))

    print(f"Best model with latent dimension {best_latent_dim} and loss {best_overall_loss} saved.")
    logging.info(f"Best model with latent dimension {best_latent_dim} and loss {best_overall_loss} saved.")

    logging.info(f'End the program at {datetime.datetime.now()}')