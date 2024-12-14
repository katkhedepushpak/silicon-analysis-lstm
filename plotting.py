# plotting.py
import torch
import matplotlib.pyplot as plt
import os
from model import LSTMModel
from sklearn.metrics import mean_squared_error

def plot_results(model_path, X_train, y_train, X_test, y_test, scaler_target, y_train1):
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=29, hidden_size=1024, num_layers=2, output_size=1).to(device)
    model.load_state_dict(torch.load(model_path))
    

    model.eval()

    # Move input tensors to the same device as the model
    X_train = X_train.to(device)
    X_test = X_test.to(device)

    # print top 10 elements from X_train, y_train, X_test, y_test
    print(X_train[:10])
    print(y_train[:10])
    print(X_test[:10])
    print(y_test[:10])
    
    with torch.no_grad():
        y_train_pred = model(X_train).cpu().numpy()
        y_pred = model(X_test).cpu().numpy()
    
    y_train1 = y_train1.cpu().numpy()
    # Inverse transform the predicted y_train_pred back to original scale
    y_train_pred_original = scaler_target.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
    
    y_test_np = y_test.cpu().numpy()
    y_pred_np = y_pred

    # Plot the predicted y_train against the actual y_train
    plt.figure(figsize=(10, 6))
    plt.plot(y_train1, label='Actual')
    plt.plot(y_train_pred_original, label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title('Actual vs Predicted Values (Training Data)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_np, label='Actual')
    plt.plot(y_pred_np, label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title('Actual vs Predicted Values (Test Data)')
    plt.legend()
    plt.show()