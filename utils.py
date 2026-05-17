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
    Calcola dinamicamente l'architettura basandosi su Feature, Classi e Dimensione del Dataset.
    L'euristica bilancia l'espansione geometrica con la prevenzione dell'overfitting.
    """
    MAX_NEURONS = 1024
    
    # 1. Regola Base di Espansione (Capacità di rappresentazione)
    # Partiamo espandendo lo spazio di input per mappare le non-linearità
    h1 = in_dim * 4
    
    # 2. Regola del Collo di Bottiglia (Sicurezza Output)
    # Il primo strato non dovrebbe mai essere inferiore al doppio delle classi
    h1 = max(h1, n_classes * 2)
    
    # 3. Regola Anti-Memorizzazione (Capacità vs Dati)
    # Se abbiamo pochi campioni, "castriamo" la rete per forzare la generalizzazione
    if num_samples < 400:
        h1 = min(64, h1)
    elif num_samples < 1500:
        h1 = min(128, h1)
        
    # 4. Fase di Distillazione (Compressione progressiva)
    # h2 comprime l'informazione, ma deve sempre essere >= n_classes
    h2 = max(n_classes, h1 // 2)
    
    # 5. Tetti Massimi di Sicurezza (Evitare colli di bottiglia computazionali)
    h1 = min(MAX_NEURONS, h1)
    h2 = min(MAX_NEURONS // 2, h2)
    
    return [int(h1), int(h2)]