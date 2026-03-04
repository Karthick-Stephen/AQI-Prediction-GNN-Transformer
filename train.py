import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys

# Import custom modules
from data.loader import load_air_quality_data
from data.preprocessing import normalize_data, handle_outliers
from models.gnn import GNNModel
from models.transformer import TransformerEncoder
from models.volatile_estimator import VolatileEstimator
from utils.metrics import evaluate_metrics

class AQIPredictionModel(nn.Module):
    """Complete AQI Prediction Model combining GNN and Transformer"""
    
    def __init__(self, input_features=21, hidden_dim=64, num_stations=14):
        super(AQIPredictionModel, self).__init__()
        self.gnn = GNNModel(input_features, hidden_dim, hidden_dim)
        self.transformer = TransformerEncoder(
            input_dim=hidden_dim,
            embed_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            forward_expansion=4,
            dropout_rate=0.2
        )
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, adjacency_matrix):
        gnn_out = self.gnn(adjacency_matrix, x)
        transformer_out = self.transformer(gnn_out.unsqueeze(0))
        output = self.fc_out(transformer_out.squeeze(0))
        return output

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """Train the AQI prediction model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch, X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch, X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    return model

def main():
    print("=" * 50)
    print("AQI Prediction with GNN and Transformer")
    print("=" * 50)
    
    print("\n[Step 1] Loading data...")
    try:
        dataset = load_air_quality_data()
        print(f"✓ Data loaded successfully")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    print("\n[Step 2] Preprocessing data...")
    try:
        df = pd.DataFrame(dataset['train'])
        df_clean = handle_outliers(df)
        print(f"✓ Outliers handled: Removed {len(df) - len(df_clean)} rows")
        
        df_normalized = normalize_data(df_clean)
        print(f"✓ Data normalized")
    except Exception as e:
        print(f"✗ Error preprocessing data: {e}")
        return
    
    print("\n[Step 3] Preparing data for training...")
    try:
        X = df_normalized.drop('aqi', axis=1).values
        y = df_normalized['aqi'].values.reshape(-1, 1)
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        print(f"✓ Train set: {len(X_train)} samples")
        print(f"✓ Val set: {len(X_val)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        return
    
    print("\n[Step 4] Initializing model...")
    try:
        model = AQIPredictionModel(input_features=X.shape[1])
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model initialized with {total_params:,} trainable parameters")
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        return
    
    print("\n[Step 5] Training model...")
    try:
        model = train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001)
        print(f"✓ Model training completed")
    except Exception as e:
        print(f"✗ Error during training: {e}")
        return
    
    print("\n[Step 6] Evaluating model...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load('models/best_model.pth'))
        model = model.to(device)
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch, X_batch)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        metrics = evaluate_metrics(np.array(all_labels), np.array(all_preds))
        
        print(f"✓ Model Evaluation Results:")
        print(f"  - RMSE: {metrics['RMSE']:.4f}")
        print(f"  - MAE: {metrics['MAE']:.4f}")
        print(f"  - R-squared: {metrics['R-squared']:.4f}")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        return
    
    print("\n" + "=" * 50)
    print("Training and evaluation completed successfully!")
    print("=" * 50)

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    main()