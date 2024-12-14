from matplotlib import rc
import math
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = "master_9_sites.csv"
master_9_sites = pd.read_csv(file_path)
master_9_sites.head(5)

# Fill NaN values
master_9_sites.fillna(value={col: 0 for col in master_9_sites.columns if col != 'Si'}, inplace=True)

# Assuming you have a dataset and dataloader setup
class ExampleDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Example data
X = master_9_sites.drop(columns=['target_column']).values  # Replace 'target_column' with your target column name
y = master_9_sites['target_column'].values  # Replace 'target_column' with your target column name

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

# Train the XGBoost model
xgb_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"XGBoost Mean Squared Error: {mse}")

# Save the plot for XGBoost model
plt.figure()
plt.plot(xgb_model.feature_importances_)
plt.title('XGBoost Feature Importances')
plt.savefig('linear_plots/xgboost_feature_importances.png')

# Assuming you have other models and their training code here
# Example for saving plots for other models
# plt.figure()
# plt.plot(other_model_results)
# plt.title('Other Model Results')
# plt.savefig('linear_plots/other_model_results.png')