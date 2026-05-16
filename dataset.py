import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

def load_uci_dataset(dataset_id: int):
    """
    Scarica il dataset e lo restituisce come DataFrame Pandas per mantenere
    i tipi di dato (numeri vs stringhe).
    """
    print(f"Scaricamento del dataset UCI ID {dataset_id}...")
    dataset = fetch_ucirepo(id=dataset_id) 
    
    X = dataset.data.features 
    y = dataset.data.targets.values.ravel()
    return X, y

def encode_labels(y):
    """
    Codifica i target testuali in valori numerici.
    Ritorna i target codificati e il numero di classi.
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    return y_encoded, num_classes

import torch
from torch.utils.data import TensorDataset, DataLoader

def create_dataloader(X, y, batch_size=64, shuffle=True):
    """
    Utility generica per convertire array Numpy in PyTorch DataLoader.
    """
    # --- SICUREZZA PYTORCH: Da Matrice Sparsa a Densa ---
    # Se il One-Hot Encoder ci manda una matrice compressa, la decomprimiamo.
    if hasattr(X, "toarray"):
        X = X.toarray()
        
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_preprocessor(X):
    # 1. Trova in automatico quali colonne sono numeri e quali sono testo
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # 2. Pipeline per i NUMERI (Riempe i buchi con la media e scala)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # 3. Pipeline per il TESTO (Riempe i buchi e fa l'One-Hot Encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 4. Unisce le due pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def get_holdout_dataloaders(X, y, batch_size=64, random_state=42):
    """
    Hold-out automatico con Preprocessor intelligente.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=random_state, stratify=y_temp
    )

    # --- PREPROCESSING UNIVERSALE ---
    # Impara (fit) solo sul Train, applica (transform) a tutto il resto
    preprocessor = get_preprocessor(X_train)
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Creazione dei DataLoader
    train_loader = create_dataloader(X_train_proc, y_train, batch_size, shuffle=True)
    val_loader = create_dataloader(X_val_proc, y_val, batch_size, shuffle=False)
    test_loader = create_dataloader(X_test_proc, y_test, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader