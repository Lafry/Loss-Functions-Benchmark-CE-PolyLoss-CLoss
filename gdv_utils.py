import torch

def calculate_gdv(X: torch.Tensor, y: torch.Tensor) -> float:
	"""
	Calcola l'indice GDV (Class Separability) date le attivazioni X e le etichette y.
	"""
	X = X.float()
	N, D = X.shape
	classes = torch.unique(y)
	L = len(classes)
	
	if L <= 1: return 0.0 
	
	mu = X.mean(dim=0)
	sigma = X.std(dim=0, unbiased=False) 
	sigma = torch.clamp(sigma, min=1e-8) 
	S = 0.5 * (X - mu) / sigma
	
	dist_matrix = torch.cdist(S, S, p=2.0)
	
	mean_intra = 0.0
	for c in classes:
		idx = (y == c)
		N_l = idx.sum().item()
		if N_l > 1:
			class_dists = dist_matrix[idx][:, idx]
			sum_dists = class_dists.sum() / 2.0
			mean_intra += (2.0 / (N_l * (N_l - 1))) * sum_dists
	mean_intra /= L
	
	mean_inter_sum = 0.0
	num_pairs = L * (L - 1) / 2
	for i in range(L):
		for j in range(i + 1, L):
			idx_i = (y == classes[i])
			idx_j = (y == classes[j])
			N_l = idx_i.sum().item()
			N_m = idx_j.sum().item()
			if N_l > 0 and N_m > 0:
				cross_dists = dist_matrix[idx_i][:, idx_j]
				mean_inter_sum += cross_dists.sum() / (N_l * N_m)
	mean_inter = mean_inter_sum / num_pairs
	
	gdv = (1.0 / torch.sqrt(torch.tensor(D, dtype=torch.float32))) * (mean_intra - mean_inter)
	return gdv.item()
