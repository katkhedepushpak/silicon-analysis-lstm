# main.py
import torch
from model import LSTMModel
from dataloader import get_data_loaders
from train import train_model
from preprocessing import load_and_preprocess_data
from plotting import plot_results
import joblib
import os

# Hyperparameters
input_size = 29  
hidden_size = 1024
num_layers = 2
output_size = 1
batch_size = 8
num_epochs = 20
learning_rate = 0.001
sequence_length = 10

# File paths
file_path = 'master_9_sites.csv'
static_file_path = 'static.csv'
model_path = 'trained_model.pth'
train_loader_path = 'train_loader.pth'
scaler_target_path = 'scaler_target.pkl'
data_processed_folder = 'data_processed'

# Ensure the data_processed folder exists
os.makedirs(data_processed_folder, exist_ok=True)

# File paths for processed data
X_train_path = os.path.join(data_processed_folder, 'X_train.pt')
y_train_path = os.path.join(data_processed_folder, 'y_train.pt')
X_test_path = os.path.join(data_processed_folder, 'X_test.pt')
y_test_path = os.path.join(data_processed_folder, 'y_test.pt')
y_train1_path = os.path.join(data_processed_folder, 'y_train1.pt')

# Load and preprocess data
X_train, y_train, X_test, y_test, scaler_target, y_train1 = load_and_preprocess_data(file_path, static_file_path, sequence_length)

# Save processed data
torch.save(X_train, X_train_path)
torch.save(y_train, y_train_path)
torch.save(X_test, X_test_path)
torch.save(y_test, y_test_path)
torch.save(y_train1, y_train1_path)
print(f"Processed data saved to {data_processed_folder}")

# Save the train_loader
train_loader = get_data_loaders(X_train, y_train, batch_size)
torch.save(train_loader, train_loader_path)
print(f"train_loader saved to {train_loader_path}")

# Save the scaler_target
joblib.dump(scaler_target, scaler_target_path)
print(f"scaler_target saved to {scaler_target_path}")

# Initialize the model
model = LSTMModel(input_size, hidden_size, num_layers, output_size).cuda()
print(model)

# Train the model
train_model(model, train_loader, num_epochs, learning_rate, model_path)
            

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
print(f"Model loaded from {model_path}")

# Fetch the train_loader from local into a new variable local_train_loader
local_train_loader = torch.load(train_loader_path)
print(f"local_train_loader loaded from {train_loader_path}")

# Fetch the scaler_target from local into a new variable local_scaler_target
local_scaler_target = joblib.load(scaler_target_path)
print(f"local_scaler_target loaded from {scaler_target_path}")

# Fetch the processed data from local
local_X_train = torch.load(X_train_path)
local_y_train = torch.load(y_train_path)
local_X_test = torch.load(X_test_path)
local_y_test = torch.load(y_test_path)
local_y_train1 = torch.load(y_train1_path, map_location=torch.device('cpu'))  # Ensure it's loaded as a tensor
print(f"Processed data loaded from {data_processed_folder}")

# Plot the results
plot_results(model_path, local_X_train, local_y_train, local_X_test, local_y_test, local_scaler_target, local_y_train1)