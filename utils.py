import json
import os
from datetime import datetime
import yaml

def salva_storico_json(history: dict, nome_esperimento: str, dataset_id: int, base_dir: str = "results"):
    """
    Salva lo storico in JSON nella struttura: results/<dataset_id>/logs/
    """
    # 1. Costruisci il percorso: results/{dataset_id}/logs
    percorso_cartella = os.path.join(base_dir, str(dataset_id), "logs")
    
    # 2. Crea la struttura ricorsivamente (crea results, poi l'id, poi logs)
    os.makedirs(percorso_cartella, exist_ok=True)
    
    # 3. Genera timestamp e nome file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nome_file = f"{timestamp}-{nome_esperimento}.json"
    filepath = os.path.join(percorso_cartella, nome_file)
    
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"[*] Storico salvato in: {filepath}")
    return filepath

def carica_storico_json(filepath: str) -> dict:
	"""
	Carica uno storico di addestramento da un file JSON.
	"""
	if not os.path.exists(filepath):
		raise FileNotFoundError(f"Il file {filepath} non esiste.")
		
	with open(filepath, 'r') as f:
		history = json.load(f)
	print(f"[*] Storico caricato da: {filepath}")
	return history

def carica_configurazione(filepath: str = "config.yaml") -> dict:
    """
    Carica i parametri di configurazione da un file YAML.
    """
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    print(f"[*] Configurazione caricata da: {filepath}")
    return config

def get_auto_hidden_layers(in_dim: int, n_classes: int, num_samples: int = 10000) -> list:
    """
    Calcola dinamicamente l'architettura basandosi su Feature e Dimensione del Dataset.
    """
    MAX_NEURONS = 1024
    
    # --- LIVELLO 1: EMERGENZA POCHI DATI (Es. Dataset 14) ---
    # Se abbiamo meno di 400 righe totali, forziamo una rete piccolissima 
    # a prescindere da quante colonne ci siano, per impedire la memorizzazione.
    if num_samples < 400:
        # Non superiamo mai i 64 neuroni nel primo strato
        h1 = min(64, in_dim * 2) 
        h2 = max(8, h1 // 2)
        
    # --- LIVELLO 2: MICRO-DATASET CLINICI (Es. Dataset 451) ---
    elif in_dim < 20:
        h1 = in_dim * 4
        h2 = h1 // 2
        
    # --- LIVELLO 3: DATASET NORMALI/GRANDI (Es. Dataset 222) ---
    else:
        MIN_NEURONS_L1 = 128
        h1 = min(MAX_NEURONS, max(MIN_NEURONS_L1, in_dim * 4))
        h2 = max(64, h1 // 2)
    
    return [int(h1), int(h2)]