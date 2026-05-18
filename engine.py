import time
import copy
import torch
import gdv_utils

def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=15, layer_for_gdv=None):
    """
    Motore di addestramento universale. Esegue il training loop, traccia le metriche,
    salva i pesi migliori in base alla validation loss e ripristina il modello a fine ciclo.
    """
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_gdv': []}
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    
    # Setup Hook GDV (universale per qualsiasi modello e layer)
    attivazioni_layer = []
    hook_handle = None
    if layer_for_gdv is not None:
        def hook_fn(module, input, output):
            attivazioni_layer.append(output.detach().cpu())
        hook_handle = layer_for_gdv.register_forward_hook(hook_fn)

    start_time = time.time()

    for epoch in range(num_epochs):
        # --- UPDATE DINAMICO C-LOSS (Soft Switching per il TRAIN) ---
        if hasattr(criterion, 'gamma'):
            # L'epoca va da 0 a num_epochs-1, calcoliamo il moltiplicatore lineare
            m = epoch
            N = max(1, num_epochs - 1)
            criterion.gamma = m / N

        # --- FASE DI TRAIN ---
        model.train()
        attivazioni_layer.clear()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        
        # --- FASE DI VALIDATION ---
        model.eval()
        val_loss, corretti, totale = 0.0, 0, 0
        attivazioni_layer.clear() 
        tutti_targets = []
        
        # --- FIX EARLY STOPPING: Il Giudice Imparziale ---
        # Salviamo il gamma corrente di training
        train_gamma = None
        if hasattr(criterion, 'gamma'):
            train_gamma = criterion.gamma
            # Valutiamo SEMPRE sulla C-Loss pura (gamma=1.0) per avere un metro fisso
            criterion.gamma = 1.0 
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                totale += targets.size(0)
                corretti += (predicted == targets).sum().item()
                tutti_targets.append(targets.cpu())
                
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * corretti / totale
        history['val_loss'].append(val_loss)
        history['val_acc'].append(accuracy)

        # Ripristiniamo il gamma dinamico per la prossima epoca di Train
        if train_gamma is not None:
            criterion.gamma = train_gamma

        # Salvataggio Modello Migliore (ora basato su una metrica fissa e coerente!)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        # Calcolo GDV (se il layer è stato fornito)
        gdv_epoca = 0.0
        if layer_for_gdv is not None and len(attivazioni_layer) > 0:
            X_gdv = torch.cat(attivazioni_layer, dim=0)
            y_gdv = torch.cat(tutti_targets, dim=0)
            if X_gdv.dim() > 2:
                X_gdv = X_gdv.view(X_gdv.size(0), -1)
            gdv_epoca = gdv_utils.calculate_gdv(X_gdv, y_gdv)
        history['val_gdv'].append(gdv_epoca)

        # Logging minimale
        if (epoch + 1) % 5 == 0 or epoch == 0:
            gdv_str = f" | Val GDV: {gdv_epoca:.4f}" if layer_for_gdv else ""
            print(f"Epoca {epoch+1:02d}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {accuracy:.2f}%{gdv_str}")

    # Pulizia Hook
    if hook_handle is not None:
        hook_handle.remove()

    print(f"\n> Training completato in {time.time() - start_time:.2f}s.")
    print(f"> Ripristino pesi dell'epoca {best_epoch} (Miglior Val Loss: {best_val_loss:.4f})")
    
    # Ripristina i pesi migliori prima di restituire il modello
    model.load_state_dict(best_model_wts)
    
    return history, best_epoch

def evaluate_model(model, dataloader, device):
    """
    Valuta un modello su un set di dati e restituisce l'accuratezza.
    Utile per il calcolo sul Test Set o sui fold finali.
    """
    model.eval()
    corretti, totale = 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            totale += targets.size(0)
            corretti += (predicted == targets).sum().item()
            
    return 100 * corretti / totale