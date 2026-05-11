import json
import os
from datetime import datetime
import yaml

def salva_storico_json(history: dict, nome_esperimento: str, dataset_id: int, base_dir: str = "results"):
    """
    Salva lo storico in JSON nella struttura: results/<dataset_id>/logs/
    """
    # 1. Costruisci il percorso: results/365/logs
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

def get_auto_hidden_layers(in_dim: int, n_classes: int) -> list:
    """
    Calcola dinamicamente un'architettura ottimale a 2 strati nascosti (Imbuto).
    Regola empirica: 
    - Layer 1 espande le feature in ingresso (per trovare combinazioni non lineari).
    - Layer 2 comprime a metà per preparare la rete alla classificazione finale.
    """
    # Definiamo dei limiti di sicurezza per evitare di esplodere la RAM 
    # o fare reti troppo stupide
    MIN_NEURONS_L1 = 128
    MAX_NEURONS = 1024
    
    # Layer 1: Moltiplica l'input per 4, ma resta tra 128 e 1024
    # (Se in_dim=16 -> max(128, 64) -> 128)
    # (Se in_dim=64 -> max(128, 256) -> 256) 
    h1 = min(MAX_NEURONS, max(MIN_NEURONS_L1, in_dim * 4))
    
    # Layer 2: Metà del primo layer
    # (Se h1=128 -> 64)
    # (Se h1=256 -> 128)
    h2 = max(64, h1 // 2)
    
    architettura = [h1, h2]
    return architettura