import torch.nn as nn

class MLP(nn.Module):
	"""
	Multi-Layer Perceptron configurabile.
	I layer e le attivazioni sono definiti come attributi espliciti per 
	facilitare l'estrazione delle feature (es. tramite Hook per il GDV).
	"""
	def __init__(self, input_size=16, hidden1=128, hidden2=64, num_classes=26, dropout_rate=0.0):
		super(MLP, self).__init__()
		
		# Primo blocco
		self.fc1 = nn.Linear(input_size, hidden1)
		self.relu1 = nn.ReLU()
		# nn.Identity() è un trucco elegante di PyTorch: se dropout_rate è 0, 
		# i dati passano attraverso questo layer senza subire alcuna modifica.
		self.drop1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
		
		# Secondo blocco (Ottimo candidato per l'estrazione del GDV)
		self.fc2 = nn.Linear(hidden1, hidden2)
		self.relu2 = nn.ReLU()
		self.drop2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
		
		# Classificatore finale
		self.fc3 = nn.Linear(hidden2, num_classes)
		
	def forward(self, x):
		x = self.drop1(self.relu1(self.fc1(x)))
		x = self.drop2(self.relu2(self.fc2(x)))
		x = self.fc3(x)
		return x