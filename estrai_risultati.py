import os
import json
import glob
import numpy as np
import pandas as pd

def estrai_riepilogo_esperimenti(base_dir="results"):
    print(f"[*] Scansione cartella '{base_dir}' alla ricerca di log JSON...")
    
    # Cerca tutti i file JSON ricorsivamente nelle cartelle dei log
    pattern = os.path.join(base_dir, "*", "logs", "*.json")
    lista_file = glob.glob(pattern)
    
    dati_holdout = []
    dati_kfold = []
    
    for filepath in lista_file:
        try:
            with open(filepath, 'r') as f:
                dati = json.load(f)
            
            nome_file = os.path.basename(filepath)

            parti_relative = os.path.relpath(filepath, base_dir).split(os.sep)
            dataset_id = parti_relative[0]  # es: "110", "198", "59", ...
            
            # --- GESTIONE LOG K-FOLD ---
            if "KFold_Results" in nome_file or "folds" in dati:
                riga_kfold = {"Dataset_ID": dataset_id, "Tipo_Esperimento": "K-Fold"}
                # Itera sulle loss presenti nel dizionario (es. Cross-Entropy, PolyLoss, C-Loss)
                for loss_name, metriche in dati.items():
                    if isinstance(metriche, dict) and "acc" in metriche and "gdv" in metriche:
                        riga_kfold[f"{loss_name}_Acc_Mean"] = np.mean(metriche["acc"])
                        riga_kfold[f"{loss_name}_Acc_Std"] = np.std(metriche["acc"])
                        riga_kfold[f"{loss_name}_GDV_Mean"] = np.mean(metriche["gdv"])
                dati_kfold.append(riga_kfold)
                
            # --- GESTIONE LOG HOLD-OUT ---
            else:
                # Cerca di capire quale loss è dal nome del file
                loss_type = "Sconosciuta"
                if "CE" in nome_file or "cross_entropy" in nome_file.lower(): loss_type = "Cross-Entropy"
                elif "PolyLoss" in nome_file: loss_type = "PolyLoss"
                elif "CombinedCLoss" in nome_file or "closs" in nome_file.lower(): loss_type = "C-Loss"
                
                # Trova la best epoch basata sulla val_loss minima
                if "val_loss" in dati and len(dati["val_loss"]) > 0:
                    best_epoch_idx = np.argmin(dati["val_loss"])
                    best_val_loss = dati["val_loss"][best_epoch_idx]
                    best_val_acc = dati["val_acc"][best_epoch_idx]
                    best_val_gdv = dati["val_gdv"][best_epoch_idx] if "val_gdv" in dati and dati["val_gdv"] else None
                    test_acc = dati.get("test_acc", None)
                    
                    dati_holdout.append({
                        "Dataset_ID": dataset_id,
                        "Loss": loss_type,
                        "Tipo_Esperimento": "Hold-Out",
                        "Best_Epoch": best_epoch_idx + 1,
                        "Best_Val_Loss": best_val_loss,
                        "Val_Acc_Best_Epoch": best_val_acc,
                        "Val_GDV_Best_Epoch": best_val_gdv,
                        "Test_Acc_Finale": test_acc
                    })
                    
        except Exception as e:
            print(f"[!] Errore nella lettura del file {filepath}: {e}")

    # Salvataggio Hold-Out
    if dati_holdout:
        df_holdout = pd.DataFrame(dati_holdout)
        df_holdout.to_csv("riepilogo_holdout.csv", index=False)
        print(f"[*] Salvato 'riepilogo_holdout.csv' con {len(df_holdout)} record.")

    # Salvataggio K-Fold
    if dati_kfold:
        df_kfold = pd.DataFrame(dati_kfold)
        df_kfold.to_csv("riepilogo_kfold.csv", index=False)
        print(f"[*] Salvato 'riepilogo_kfold.csv' con {len(df_kfold)} record.")

if __name__ == "__main__":
    estrai_riepilogo_esperimenti()