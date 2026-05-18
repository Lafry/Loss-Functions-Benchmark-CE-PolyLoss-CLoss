import torch
import torch.nn as nn
import torch.nn.functional as F

class PolyLoss(nn.Module):
    """
    Implementazione della PolyLoss (Poly-1).
    Formula: L_Poly = L_CE + epsilon * (1 - p_t)
    
    Aggiunge un termine polinomiale alla Cross-Entropy per spingere 
    il modello a concentrarsi non solo sugli esempi difficili, ma 
    anche a migliorare la confidenza sulle predizioni corrette.
    """
    def __init__(self, epsilon: float = 2.0, reduction: str = 'mean', weight: torch.Tensor = None):
        super(PolyLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        # Iniettiamo il tensore dei pesi (se fornito) per bilanciare le classi
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calcolo della Cross-Entropy non ridotta (vettore di loss)
        ce_loss = self.ce(outputs, targets)
        
        # Calcolo della probabilità predetta per la vera classe (p_t)
        pt = F.softmax(outputs, dim=1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Applicazione della formula Poly-1
        poly_loss = ce_loss + self.epsilon * (1.0 - pt)
        
        # Gestione della riduzione
        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        return poly_loss


class CombinedCLoss(nn.Module):
    """
    Implementazione della Combined C-Loss.
    Combina la Cross-Entropy standard con una Correntropy-induced Loss (C-Loss).
    Questo approccio ibrido mira a migliorare la robustezza del modello contro 
    outlier e rumore nei dati, bilanciando l'apprendimento globale e locale.
    """
    def __init__(self, sigma: float = 0.5, reduction: str = 'mean', weight: torch.Tensor = None):
        super(CombinedCLoss, self).__init__()
        self.sigma = sigma
        self.gamma = 0.0  # Verrà aggiornato dinamicamente dall'engine (Soft-Switch)
        self.reduction = reduction
        # Iniettiamo il tensore dei pesi (se fornito) per bilanciare le classi
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(outputs, targets)
        c_loss = 1.0 - torch.exp(- (ce_loss ** 2) / (2 * self.sigma ** 2))
        
        # Combinazione dinamica secondo Singh et al. (Eq. 30)
        combined_loss = (1.0 - self.gamma) * ce_loss + self.gamma * c_loss
        
        if self.reduction == 'mean':
            return combined_loss.mean()
        elif self.reduction == 'sum':
            return combined_loss.sum()
        return combined_loss