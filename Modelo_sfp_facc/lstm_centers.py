"""
lstm_centers.py
===============
Réplica de Sec. 2.5.1 del paper Qin et al. (2024).

Ajusta los centros de cluster mediante un modelo LSTM por clase, evitando
el costo computacional de actualizar centros iterativamente durante el
proceso de clasificación del ACO.

Hiperparámetros del paper (Tabla 3):
- Input window size: 40
- Dropout: 0.1
- Loss: MSE
- Learning rate: 0.001
- Epochs: 200
- Batch size: 4

Decisión de implementación: el paper es ambiguo sobre qué predice el LSTM. 
Interpretamos "fitting cluster centers" como entrenar un modelo que aprende 
a generar la señal-prototipo de cada clase. Implementación: por cada clase,
entrenamos un LSTM secuencia-a-secuencia donde el input es la secuencia 
desplazada y el output es la siguiente muestra (autoregresión); luego 
generamos el centro corriendo el modelo recursivamente desde la media inicial 
de la clase. Esto coincide con la Fig. 4 del paper donde la "fitting result" 
parece una versión suavizada de las señales de entrenamiento.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =====================================================================
# Configuración (Tabla 3 del paper)
# =====================================================================

LSTM_CONFIG = {
    'input_window': 40,
    'hidden_size': 64,    # paper no especifica; default razonable
    'num_layers': 1,      # paper no especifica; mantenemos simple
    'dropout': 0.1,
    'learning_rate': 0.001,
    'epochs': 200,
    'batch_size': 4,
}


# =====================================================================
# Modelo LSTM (Eqs. 10-16 del paper)
# =====================================================================

class ECGLSTM(nn.Module):
    """
    LSTM para regresión autoregresiva de señales ECG.
    
    Implementa las gates descritas en las Eqs. 10-16 del paper:
    - Forget gate (Eq. 10-11)
    - Input gate (Eq. 12)
    - Cell update (Eq. 13-14)
    - Output gate (Eq. 15-16)
    
    PyTorch ya implementa estas gates internamente en nn.LSTM.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size=1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # último paso
        return self.fc(out).squeeze(-1)  # (batch,)


# =====================================================================
# Dataset
# =====================================================================

class WindowedECGDataset(Dataset):
    """
    Construye ventanas deslizantes de las señales de fitting.
    Para cada señal de longitud L, genera (L - window) pares (input, target):
        input  = señal[i : i+window]
        target = señal[i+window]
    """
    def __init__(self, signals: np.ndarray, window_size: int = 40):
        self.window_size = window_size
        self.inputs = []
        self.targets = []
        for sig in signals:
            for i in range(len(sig) - window_size):
                self.inputs.append(sig[i:i + window_size])
                self.targets.append(sig[i + window_size])
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).unsqueeze(-1)  # (window, 1)
        y = torch.tensor(self.targets[idx])
        return x, y


# =====================================================================
# Entrenamiento por clase
# =====================================================================

def train_class_lstm(class_signals: np.ndarray,
                      config: dict = None,
                      verbose: bool = True) -> ECGLSTM:
    """
    Entrena un LSTM autoregresivo sobre las señales de una clase.
    
    Parámetros:
    -----------
    class_signals : array (n_signals, signal_length) - señales de la clase
    config : dict - hiperparámetros (default LSTM_CONFIG)
    
    Devuelve:
    ---------
    Modelo LSTM entrenado.
    """
    if config is None:
        config = LSTM_CONFIG
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = WindowedECGDataset(class_signals, window_size=config['input_window'])
    loader = DataLoader(dataset,
                        batch_size=config['batch_size'],
                        shuffle=True)
    
    model = ECGLSTM(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()  # paper dice "Lose Function: mse"
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{config['epochs']}, "
                  f"Loss: {total_loss / len(dataset):.6f}")
    
    return model


# =====================================================================
# Generación del centro de cluster a partir del LSTM entrenado
# =====================================================================

def generate_cluster_center(model: ECGLSTM,
                             seed_signal: np.ndarray,
                             target_length: int,
                             window_size: int = 40) -> np.ndarray:
    """
    Genera la señal-prototipo (centro de cluster) corriendo el LSTM 
    autoregresivamente.
    
    Parámetros:
    -----------
    model : LSTM entrenado
    seed_signal : ventana inicial (típicamente la media de las señales de la clase)
    target_length : longitud deseada del centro generado
    window_size : tamaño de la ventana de entrada del LSTM
    
    Devuelve:
    ---------
    Señal generada de longitud target_length.
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Inicializar con seed
    generated = list(seed_signal[:window_size].astype(np.float32))
    
    with torch.no_grad():
        for _ in range(target_length - window_size):
            window = np.array(generated[-window_size:], dtype=np.float32)
            x = torch.from_numpy(window).unsqueeze(0).unsqueeze(-1).to(device)
            next_val = model(x).item()
            generated.append(next_val)
    
    return np.array(generated)


def fit_all_cluster_centers(class_signals_dict: dict,
                              fit_n_per_class: int = 10,
                              config: dict = None,
                              verbose: bool = True,
                              seed: int = 42) -> dict:
    """
    Entrena un LSTM por clase y genera los centros de cluster correspondientes.
    
    Parámetros:
    -----------
    class_signals_dict : dict {class_name: array (n_signals, signal_length)}
    fit_n_per_class : número de señales aleatorias por clase para fitting (paper: 10)
    config : hiperparámetros LSTM
    seed : semilla para reproducibilidad
    
    Devuelve:
    ---------
    centers_dict : {class_name: array (signal_length,)}
    """
    rng = np.random.default_rng(seed)
    centers = {}
    
    for class_name, signals in class_signals_dict.items():
        if verbose:
            print(f"\n[LSTM] Fitting centro para clase '{class_name}'...")
        
        # Selección aleatoria de fit_n_per_class señales
        n_available = len(signals)
        n_to_use = min(fit_n_per_class, n_available)
        idx = rng.choice(n_available, size=n_to_use, replace=False)
        fit_signals = signals[idx]
        
        if verbose:
            print(f"  Usando {n_to_use} señales (paper sugiere 10)")
        
        # Entrenar
        model = train_class_lstm(fit_signals, config=config, verbose=verbose)
        
        # Generar centro
        signal_length = signals.shape[1]
        seed_signal = fit_signals.mean(axis=0)  # promedio como seed
        window_size = (config or LSTM_CONFIG)['input_window']
        center = generate_cluster_center(model, seed_signal, signal_length, window_size)
        
        centers[class_name] = center
    
    return centers


if __name__ == "__main__":
    # Smoke test mínimo (epochs=5)
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Crear señales sintéticas para 2 clases
    fake_class_a = np.array([
        np.sin(np.linspace(0, 4 * np.pi, 250)) + 0.1 * np.random.randn(250)
        for _ in range(15)
    ])
    fake_class_b = np.array([
        np.cos(np.linspace(0, 4 * np.pi, 250)) + 0.1 * np.random.randn(250)
        for _ in range(15)
    ])
    
    config_test = LSTM_CONFIG.copy()
    config_test['epochs'] = 5  # rápido para smoke test
    
    centers = fit_all_cluster_centers(
        {'A': fake_class_a, 'B': fake_class_b},
        fit_n_per_class=10,
        config=config_test,
    )
    print(f"\nCentros generados: {list(centers.keys())}")
    print(f"Tamaño centro A: {centers['A'].shape}")
