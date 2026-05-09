import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers=[128, 64], dropout_rate=0.0):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        in_features = input_size
        
        # Costruisce la rete in base alla lista in config.yaml
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(in_features, hidden_dim))
            self.activations.append(nn.ReLU())
            self.dropouts.append(nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity())
            in_features = hidden_dim
            
        # Classificatore finale
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        for layer, act, drop in zip(self.layers, self.activations, self.dropouts):
            x = drop(act(layer(x)))
        x = self.classifier(x)
        return x

    def get_layer_for_gdv(self, index=-1):
        """
        Restituisce lo strato di attivazione (ReLU) specificato per estrarre il GDV.
        Di default (index=-1) restituisce l'attivazione dell'ultimo strato nascosto.
        """
        return self.activations[index]