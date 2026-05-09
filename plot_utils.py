import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==========================================
# CONFIGURAZIONE GLOBALE PER STAMPA TESI (LaTeX)
# ==========================================
plt.rcParams.update({
	'font.family': 'serif', 	# Usa font Serif per abbinarsi a LaTeX
	'font.size': 11, 			# Dimensione base
	'axes.titlesize': 13, 		# Titoli grafici
	'axes.labelsize': 11, 		# Etichette X, Y
	'xtick.labelsize': 10, 	    # Numeri asse X
	'ytick.labelsize': 10, 	    # Numeri asse Y
	'legend.fontsize': 10, 	    # Testo legenda
	'lines.linewidth': 1.0,     # Linee più spesse per la stampa
	'lines.markersize': 8, 	    # Marker ben visibili
	'figure.dpi': 300, 		    # Alta risoluzione
})

def plot_training_history(history: dict, model_name: str = "Modello", gdv_iniziale: float = None, save_pdf: bool = True, save_dir: str = "plots"):
	"""
	Disegna i grafici di Loss, Accuratezza e (se presente) GDV in base allo storico.
	Ottimizzato per documenti LaTeX: linee pulite, marker solo per il GDV iniziale.
	Salva i file in formato PDF con timestamp nella cartella specificata.
	"""
	epoche = list(range(1, len(history['train_loss']) + 1))
	
	# Puliamo il nome del modello per generare nomi file validi per il salvataggio
	safe_model_name = model_name.replace(' ', '_').replace('/', '-')

	# Creazione della cartella e del prefisso con timestamp
	if save_pdf:
		os.makedirs(save_dir, exist_ok=True)
		timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		base_filename = f"{timestamp}_{safe_model_name}"

	# Definizione colori ottimizzati per stampa (più scuri e saturi)
	C_TRAIN = '#004488' # Deep Blue
	C_VAL = '#BB5500'   # Burnt Orange
	C_ACC = '#117733'   # Forest Green
	C_GDV = '#332288'   # Deep Purple
	C_TEST = '#990000'  # Dark Red
	
	# ==========================================
	# GRAFICO 1: LOSS
	# ==========================================
	plt.figure(figsize=(6, 3.5))
	
	# Linee fluide senza marker
	plt.plot(epoche, history['train_loss'], label='Train Loss', color=C_TRAIN)
	plt.plot(epoche, history['val_loss'], label='Validation Loss', color=C_VAL)
	
	plt.title(f'{model_name} - Andamento Loss')
	plt.xlabel('Epoche')
	plt.ylabel('Loss')
	plt.legend()
	plt.grid(True, linestyle='--', alpha=0.7)
	plt.tight_layout()
	
	if save_pdf:
		filepath = os.path.join(save_dir, f"{base_filename}_Loss.pdf")
		plt.savefig(filepath, format='pdf', bbox_inches='tight')
		print(f"[*] Grafico salvato: {filepath}")
		
	plt.show() 
	
	# ==========================================
	# GRAFICO 2: ACCURATEZZA
	# ==========================================
	plt.figure(figsize=(6, 3.5))
	
	# Linea fluida senza marker
	plt.plot(epoche, history['val_acc'], label='Validation Accuracy', color=C_ACC)
	
	if 'test_acc' in history:
		plt.axhline(y=history['test_acc'], color=C_TEST, linestyle='--', label=f"Test Finale ({history['test_acc']:.2f}%)")
		
	plt.title(f'{model_name} - Andamento Accuratezza')
	plt.xlabel('Epoche')
	plt.ylabel('Accuratezza (%)')
	plt.legend(loc='lower right') # Posizionato solitamente in basso a destra per l'accuratezza
	plt.grid(True, linestyle='--', alpha=0.7)
	plt.tight_layout()
	
	if save_pdf:
		filepath = os.path.join(save_dir, f"{base_filename}_Accuracy.pdf")
		plt.savefig(filepath, format='pdf', bbox_inches='tight')
		print(f"[*] Grafico salvato: {filepath}")
		
	plt.show() 
	
	# ==========================================
	# GRAFICO 3: GDV (Opzionale)
	# ==========================================
	ha_gdv = 'val_gdv' in history and len(history['val_gdv']) > 0 and any(v != 0.0 for v in history['val_gdv'])
	
	if ha_gdv:
		gdv_values = list(history['val_gdv'])
		epoche_gdv = epoche.copy()
		
		# Se abbiamo il GDV grezzo, lo inseriamo all'Epoca 0
		if gdv_iniziale is not None:
			gdv_values = [gdv_iniziale] + gdv_values
			epoche_gdv = [0] + epoche_gdv
			
		plt.figure(figsize=(6, 3.5))
		
		# Linea fluida senza marker per le epoche successive
		plt.plot(epoche_gdv, gdv_values, label='Validation GDV', color=C_GDV)
		
		# Evidenziamo ESCLUSIVAMENTE il punto di partenza (Epoca 0) con un marker
		if gdv_iniziale is not None:
			plt.plot(0, gdv_iniziale, marker='o', color='black', markersize=6, 
					linestyle='None', label='GDV Iniziale (Grezzo)')

		plt.title(f'{model_name} - Class Separability (GDV)')
		plt.xlabel('Epoche')
		plt.ylabel('GDV (più basso è meglio)')

		# Dinamismo dell'asse Y con limiti base -1.0 e 0.0
		min_gdv = min(gdv_values)
		max_gdv = max(gdv_values)
		
		y_bottom = min(-1.0, min_gdv)
		y_top = max(0.0, max_gdv)
		
		plt.ylim(bottom=y_bottom, top=y_top)

		plt.legend()
		plt.grid(True, linestyle='--', alpha=0.7)
		plt.tight_layout()
		
		if save_pdf:
			filepath = os.path.join(save_dir, f"{base_filename}_GDV.pdf")
			plt.savefig(filepath, format='pdf', bbox_inches='tight')
			print(f"[*] Grafico salvato: {filepath}")
			
		plt.show()