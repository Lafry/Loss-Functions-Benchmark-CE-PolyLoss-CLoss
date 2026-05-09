import torch

def calculate_gdv(X: torch.Tensor, y: torch.Tensor) -> float:
    """
    Calcola l'indice GDV (Class Separability) date le attivazioni X e le etichette y.
    Versione OTTIMIZZATA PER LA MEMORIA: calcola le distanze on-the-fly per 
    evitare l'esplosione della RAM (Out Of Memory) su dataset di grandi dimensioni.
    """
    X = X.float()
    N, D = X.shape
    classes = torch.unique(y)
    L = len(classes)
    
    if L <= 1: return 0.0 
    
    # Standardizzazione geometrica
    mu = X.mean(dim=0)
    sigma = X.std(dim=0, unbiased=False) 
    sigma = torch.clamp(sigma, min=1e-8) 
    S = 0.5 * (X - mu) / sigma
    
    # Calcolo Distanze Intra-Classe (Distanza tra elementi della stessa classe)
    mean_intra = 0.0
    for c in classes:
        idx = (y == c)
        S_c = S[idx] # Estraiamo solo i punti di questa classe
        N_l = S_c.size(0)
        
        if N_l > 1:
            # Calcoliamo la matrice di distanza solo per questo sottoinsieme (Molto più piccolo!)
            class_dists = torch.cdist(S_c, S_c, p=2.0)
            sum_dists = class_dists.sum() / 2.0
            mean_intra += (2.0 / (N_l * (N_l - 1))) * sum_dists
    mean_intra /= L
    
    # Calcolo Distanze Inter-Classe (Distanza tra elementi di classi diverse)
    mean_inter_sum = 0.0
    num_pairs = L * (L - 1) / 2
    for i in range(L):
        for j in range(i + 1, L):
            idx_i = (y == classes[i])
            idx_j = (y == classes[j])
            
            S_i = S[idx_i]
            S_j = S[idx_j]
            
            N_l = S_i.size(0)
            N_m = S_j.size(0)
            
            if N_l > 0 and N_m > 0:
                # Calcoliamo la distanza incrociata solo tra questi due sottoinsiemi
                cross_dists = torch.cdist(S_i, S_j, p=2.0)
                mean_inter_sum += cross_dists.sum() / (N_l * N_m)
    mean_inter = mean_inter_sum / num_pairs
    
    # Formula finale GDV
    gdv = (1.0 / torch.sqrt(torch.tensor(D, dtype=torch.float32))) * (mean_intra - mean_inter)
    return gdv.item()