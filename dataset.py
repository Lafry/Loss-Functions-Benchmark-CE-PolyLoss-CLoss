import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo

def load_uci_dataset(dataset_id: int):
    """
    Scarica il dataset in base all'ID UCI fornito.
    """
    print(f"Scaricamento del dataset UCI ID {dataset_id}...")
    dataset = fetch_ucirepo(id=dataset_id) 
    X = dataset.data.features.values
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

def create_dataloader(X, y, batch_size=64, shuffle=True):
    """
    Utility generica per convertire array Numpy in PyTorch DataLoader.
    """
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_holdout_dataloaders(X, y, batch_size=64, random_state=42):
    """
    Esegue lo split per l'approccio Hold-out (Train, Val, Test),
    risolve i valori mancanti (NaN) e applica lo scaling 
    evitando il Data Leakage.
    """
    # Split 1: Isola il Test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state, stratify=y
    )
    
    # Split 2: Isola Train e Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=random_state, stratify=y_temp
    )

    # --- FIX: IMPUTAZIONE VALORI MANCANTI (NaN) ---
    # Il .fit() si fa SOLO sul Train set. Val e Test subiscono solo .transform()
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    # --- SCALING ---
    # Anche qui, il .fit() si fa SOLO sul Train set già imputato
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Creazione dei DataLoader
    train_loader = create_dataloader(X_train_scaled, y_train, batch_size, shuffle=True)
    val_loader = create_dataloader(X_val_scaled, y_val, batch_size, shuffle=False)
    test_loader = create_dataloader(X_test_scaled, y_test, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader