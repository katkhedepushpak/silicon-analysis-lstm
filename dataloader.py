# data_loader.py
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_data_loaders(X_train, y_train, batch_size):
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(train_dataset)
    print(train_loader)
    return train_loader