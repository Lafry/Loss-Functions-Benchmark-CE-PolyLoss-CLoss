import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==========================================
# CONFIGURAZIONE GLOBALE PER STAMPA TESI (LaTeX)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',     # Usa font Serif per abbinarsi a LaTeX
    'font.size': 11,            # Dimensione base
    'axes.titlesize': 13,       # Titoli grafici
    'axes.labelsize': 11,       # Etichette X, Y
    'xtick.labelsize': 10,      # Numeri asse X
    'ytick.labelsize': 10,      # Numeri asse Y
    'legend.fontsize': 10,      # Testo legenda
    'lines.linewidth': 1.0,     # Linee più spesse per la stampa
    'lines.markersize': 8,      # Marker ben visibili
    'figure.dpi': 300,          # Alta risoluzione
})

def plot_training_history(history: dict, model_name: str = "Modello", gdv_iniziale: float = None, 
                          save_pdf: bool = True, dataset_id: int = None, base_dir: str = "results",
                          y_limits: dict = None): # <--- Nuovo parametro
    """
    y_limits: dizionario con chiavi 'loss', 'acc', 'gdv' contenenti tuple (min, max)
    """
    epoche = list(range(1, len(history['train_loss']) + 1))
    safe_model_name = model_name.replace(' ', '_').replace('/', '-')

    if save_pdf:
        folder_id = str(dataset_id) if dataset_id is not None else "generic"
        percorso_cartella = os.path.join(base_dir, folder_id, "plots")
        os.makedirs(percorso_cartella, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_filename = f"{timestamp}_{safe_model_name}"

    C_TRAIN, C_VAL, C_ACC, C_GDV, C_TEST = '#004488', '#BB5500', '#117733', '#332288', '#990000'
    
    # --- GRAFICO 1: LOSS ---
    plt.figure(figsize=(6, 3.5))
    plt.plot(epoche, history['train_loss'], label='Train Loss', color=C_TRAIN)
    plt.plot(epoche, history['val_loss'], label='Validation Loss', color=C_VAL)
    if y_limits and 'loss' in y_limits:
        plt.ylim(y_limits['loss']) # <--- Applica scala globale
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoche'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    if save_pdf: plt.savefig(os.path.join(percorso_cartella, f"{base_filename}_Loss.pdf"), format='pdf', bbox_inches='tight')
    plt.show() 

    # --- GRAFICO 2: ACCURATEZZA ---
    plt.figure(figsize=(6, 3.5))
    plt.plot(epoche, history['val_acc'], label='Val Acc', color=C_ACC)
    if 'test_acc' in history:
        plt.axhline(y=history['test_acc'], color=C_TEST, linestyle='--', label=f"Test ({history['test_acc']:.2f}%)")
    if y_limits and 'acc' in y_limits:
        plt.ylim(y_limits['acc']) # <--- Applica scala globale
    plt.title(f'{model_name} - Accuratezza')
    plt.xlabel('Epoche'); plt.ylabel('Acc (%)'); plt.legend(loc='lower right'); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    if save_pdf: plt.savefig(os.path.join(percorso_cartella, f"{base_filename}_Accuracy.pdf"), format='pdf', bbox_inches='tight')
    plt.show() 

    # --- GRAFICO 3: GDV ---
    ha_gdv = 'val_gdv' in history and len(history['val_gdv']) > 0 and any(v != 0.0 for v in history['val_gdv'])
    if ha_gdv:
        gdv_values = [gdv_iniziale] + list(history['val_gdv']) if gdv_iniziale is not None else list(history['val_gdv'])
        epoche_gdv = [0] + epoche if gdv_iniziale is not None else epoche
        plt.figure(figsize=(6, 3.5))
        plt.plot(epoche_gdv, gdv_values, label='Validation GDV', color=C_GDV)
        if gdv_iniziale is not None:
            plt.plot(0, gdv_iniziale, marker='o', color='black', markersize=6, linestyle='None', label='Grezzo')
        
        if y_limits and 'gdv' in y_limits:
            plt.ylim(y_limits['gdv']) # <--- Applica scala globale
        else:
            plt.ylim(bottom=min(-1.0, min(gdv_values)), top=max(0.0, max(gdv_values)))
            
        plt.title(f'{model_name} - GDV'); plt.xlabel('Epoche'); plt.ylabel('GDV'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
        if save_pdf: plt.savefig(os.path.join(percorso_cartella, f"{base_filename}_GDV.pdf"), format='pdf', bbox_inches='tight')
        plt.show()