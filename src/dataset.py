import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import DATA_FILE, TEST_SPLIT, RANDOM_STATE, BATCH_SIZE

class WaterDataset(Dataset):
    """PyTorch Dataset for water potability data."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_prep_data():
    """Loads CSV, imputes missing values, splits and scales the data."""
    df = pd.read_csv(DATA_FILE)
    
    # Fill missing values with the column average
    df = df.fillna(df.mean())
    
    # Separate features and target
    X = df.drop('Potability', axis=1).values
    y = df['Potability'].values
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE
    )
    
    # Scale features (critical for neural networks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, df

def get_dataloaders():
    """Returns training and testing DataLoaders along with the raw dataframe for viz."""
    X_train, X_test, y_train, y_test, df = load_and_prep_data()
    
    train_ds = WaterDataset(X_train, y_train)
    test_ds = WaterDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, df
