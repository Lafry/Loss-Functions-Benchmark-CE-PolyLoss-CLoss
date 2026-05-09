import json
import os
from datetime import datetime

def salva_storico_json(history: dict, nome_esperimento: str, directory: str = "logs"):
	"""
	Salva il dizionario dello storico in JSON aggiungendo data e ora al nome del file.
	Esempio: logs/2026-05-04_14-30-00-holdout_cross_entropy.json
	"""
	# Crea la cartella se non esiste
	os.makedirs(directory, exist_ok=True)
	
	# Genera il timestamp attuale (Anno-Mese-Giorno_Ora-Minuto-Secondo)
	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	
	# Costruisci il nome del file finale
	nome_file = f"{timestamp}-{nome_esperimento}.json"
	filepath = os.path.join(directory, nome_file)
	
	with open(filepath, 'w') as f:
		json.dump(history, f, indent=4)
		
	print(f"[*] Storico salvato in: {filepath}")
	return filepath # Restituisce il percorso per poterlo stampare o riutilizzare

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