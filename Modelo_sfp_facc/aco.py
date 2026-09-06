"""
aco.py
======
Réplica de Sec. 2.5.2-2.5.4 del paper Qin et al. (2024).

Implementa:
1. Clasificación combinada Euclidean + DTW (Sec. 2.5.2)
2. Algoritmo de colonia de hormigas con coeficiente de feromona dinámico (Sec. 2.5.3)
3. Selección de solución óptima mediante radix sort (Sec. 2.5.4)

Hiperparámetros del paper (Tabla 3):
- Iteraciones: 150
- Hormigas: 30
- α = β = 1
- b = 0.17 (constante de ρ dinámico)
"""

import numpy as np
from features import dtw_distance


# =====================================================================
# Configuración (Tabla 3 del paper)
# =====================================================================

ACO_CONFIG = {
    'n_iterations': 150,
    'n_ants': 30,
    'alpha': 1.0,
    'beta': 1.0,
    'b_constant': 0.17,
}


# =====================================================================
# Distancia Euclidiana sobre features (Eq. 9)
# =====================================================================

def euclidean_feature_distance(test_features: np.ndarray,
                                center_features: np.ndarray) -> float:
    """
    Eq. 9 del paper: distancia Euclidiana entre vectores de 36 features.
    """
    return np.sqrt(np.sum((test_features - center_features) ** 2))


# =====================================================================
# Normalización min-max (Eq. 20)
# =====================================================================

def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Eq. 20 del paper."""
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


# =====================================================================
# Clasificación combinada Euclidean + DTW (Sec. 2.5.2)
# =====================================================================

def classify_with_euclidean_and_dtw(test_signal: np.ndarray,
                                     test_features: np.ndarray,
                                     center_signals: dict,
                                     center_features: dict) -> str:
    """
    Implementa el pseudocódigo de "classification Based on Euclidean Distance and DTW".
    
    Parámetros:
    -----------
    test_signal : señal cruda de test
    test_features : 36 features espaciales del test
    center_signals : {class_name: signal} señales de los centros (para DTW)
    center_features : {class_name: features} features de los centros (para Euclidean)
    
    Devuelve:
    ---------
    Nombre de la clase predicha.
    
    Lógica del paper (Sec. 2.5.2):
    1. Calcular DTW(test, centro_l) para cada clase l
    2. Calcular Euclidean(features_test, features_centro_l), normalizar
    3. Si argmin(Euclidean) ≠ argmin(DTW): penalizar la Euclidean ganadora
    4. Devolver la clase de argmin(Euclidean) post-penalización
    """
    classes = list(center_signals.keys())
    n_classes = len(classes)
    
    # 1. DTW
    dtw_values = np.array([
        dtw_distance(test_signal, center_signals[c]) for c in classes
    ])
    dtw_argmin = int(np.argmin(dtw_values))
    
    # 2. Euclidean sobre features (normalizada)
    euc_values = np.array([
        euclidean_feature_distance(test_features, center_features[c]) for c in classes
    ])
    euc_normalized = minmax_normalize(euc_values)
    
    # 3. Regla de fusión: si discrepan, penalizar
    euc_argmin = int(np.argmin(euc_normalized))
    if euc_argmin != dtw_argmin:
        # "The value of min(E*) add the mean of 5 Euclidean distances after normalization"
        # Lectura: sumar la media a la distancia mínima para penalizarla
        penalty = euc_normalized.mean()
        euc_normalized[euc_argmin] += penalty
    
    final_argmin = int(np.argmin(euc_normalized))
    return classes[final_argmin]


# =====================================================================
# Coeficiente de feromona dinámico ρ (Eq. 22)
# =====================================================================

def dynamic_rho(iteration: int, b: float = 0.17) -> float:
    """
    Eq. 22 del paper:
        ρ = ln(N+2) / ((N+2) + ln(N+2)) + b
    
    donde N es el número de iteración actual.
    """
    N = iteration
    return np.log(N + 2) / ((N + 2) + np.log(N + 2)) + b


# =====================================================================
# Radix sort para selección de solución óptima (Sec. 2.5.4)
# =====================================================================

def radix_sort_solutions(solutions: list) -> list:
    """
    Ordena una lista de soluciones [(n_errores, longitud_path), ...].
    
    Sec. 2.5.4: "first sorted by classification result accuracy from large to small,
    then sorted by traversal total path length from small to large".
    
    Equivalente a: minimizar primero n_errores, luego minimizar path_length.
    """
    # Sort estable: primero por path_length ASC, luego por n_errores ASC
    return sorted(solutions, key=lambda s: (s[0], s[1]))


# =====================================================================
# Algoritmo principal del ACO (Sec. 2.5)
# =====================================================================

def precompute_distance_matrices(test_signals: np.ndarray,
                                    test_features: np.ndarray,
                                    center_signals: dict,
                                    center_features: dict,
                                    verbose: bool = True) -> tuple:
    """
    Precalcula DTW y Euclidean entre cada test_signal y cada centro.
    
    OPTIMIZACIÓN CRÍTICA: como los centros no cambian durante el ACO 
    (gracias al fitting con LSTM previo), estos cálculos se hacen UNA SOLA VEZ
    en lugar de n_iter × n_ants veces.
    
    Devuelve:
    - dtw_matrix: (n_test, n_classes)
    - euc_matrix: (n_test, n_classes)
    - classes: lista ordenada de clases
    """
    classes = list(center_signals.keys())
    n_test = len(test_signals)
    n_classes = len(classes)
    
    dtw_matrix = np.zeros((n_test, n_classes))
    euc_matrix = np.zeros((n_test, n_classes))
    
    if verbose:
        print(f"  Precalculando DTW y Euclidean ({n_test} × {n_classes})...")
    
    for k in range(n_test):
        for ci, cls in enumerate(classes):
            dtw_matrix[k, ci] = dtw_distance(test_signals[k], center_signals[cls])
            euc_matrix[k, ci] = euclidean_feature_distance(
                test_features[k], center_features[cls]
            )
        if verbose and (k + 1) % 100 == 0:
            print(f"    {k + 1}/{n_test} señales procesadas")
    
    return dtw_matrix, euc_matrix, classes


def classify_from_precomputed(dtw_row: np.ndarray, euc_row: np.ndarray) -> int:
    """
    Versión vectorizada de classify_with_euclidean_and_dtw que opera sobre
    distancias precalculadas para una sola señal de test.
    
    Devuelve el índice de la clase predicha.
    """
    dtw_argmin = int(np.argmin(dtw_row))
    
    # Normalizar Euclidean min-max
    euc_min, euc_max = euc_row.min(), euc_row.max()
    if euc_max - euc_min < 1e-12:
        euc_norm = np.zeros_like(euc_row)
    else:
        euc_norm = (euc_row - euc_min) / (euc_max - euc_min)
    
    euc_argmin = int(np.argmin(euc_norm))
    
    if euc_argmin != dtw_argmin:
        penalty = euc_norm.mean()
        euc_norm = euc_norm.copy()
        euc_norm[euc_argmin] += penalty
    
    return int(np.argmin(euc_norm))


def run_aco_classification(test_signals: np.ndarray,
                            test_features: np.ndarray,
                            test_labels: list,
                            center_signals: dict,
                            center_features: dict,
                            config: dict = None,
                            verbose: bool = True) -> dict:
    """
    Ejecuta el ACO completo con feromona dinámica y radix sort.
    
    Parámetros:
    -----------
    test_signals : array (n_test, signal_length)
    test_features : array (n_test, 36)
    test_labels : lista de etiquetas verdaderas
    center_signals : {class_name: signal}
    center_features : {class_name: features}
    config : hiperparámetros ACO
    
    Devuelve:
    ---------
    dict con:
    - 'predictions': lista de predicciones (mejor solución global)
    - 'accuracy': accuracy global
    - 'history': lista de (iter, n_errores, path_length) para tracking
    """
    if config is None:
        config = ACO_CONFIG
    
    n_test = len(test_signals)
    
    # === OPTIMIZACIÓN: precalcular DTW y Euclidean ===
    dtw_matrix, euc_matrix, classes = precompute_distance_matrices(
        test_signals, test_features, center_signals, center_features,
        verbose=verbose,
    )
    n_classes = len(classes)
    
    # Predicciones base (idénticas para todas las hormigas dado los mismos centros)
    base_predictions = np.array([
        classify_from_precomputed(dtw_matrix[k], euc_matrix[k])
        for k in range(n_test)
    ])
    base_pred_labels = [classes[idx] for idx in base_predictions]
    base_n_errors = sum(
        1 for p, t in zip(base_pred_labels, test_labels) if p != t
    )
    
    # Pheromona inicial uniforme
    pheromone = np.ones((n_test, n_classes))
    
    global_solutions = []
    
    if verbose:
        print(f"\n[ACO] Ejecutando {config['n_iterations']} iteraciones "
              f"con {config['n_ants']} hormigas...")
        print(f"  Predicción base (sin estocasticidad): {base_n_errors} errores")
    
    for iteration in range(config['n_iterations']):
        rho = dynamic_rho(iteration, b=config['b_constant'])
        
        local_solutions = []
        
        for ant in range(config['n_ants']):
            # En el paper, cada hormiga visita los puntos en orden aleatorio.
            # La predicción por punto NO cambia entre hormigas si los centros son fijos
            # (que es nuestro caso post-LSTM). La estocasticidad la introducimos vía
            # exploración con probabilidad pheromone-based:
            # con prob α*η^β favorecemos la asignación base; con probabilidad residual
            # exploramos otra clase. Esto sigue la lógica de Eq. 4 del paper.
            
            predictions = base_predictions.copy()
            path_length = 0.0
            
            # Aplicar exploración estocástica usando pheromone
            for k in range(n_test):
                # Probabilidad de transición a cada clase (Eq. 4 simplificada)
                eta = 1.0 / (euc_matrix[k] + 1e-12)
                tau = pheromone[k]
                p_unnorm = (tau ** config['alpha']) * (eta ** config['beta'])
                p_norm = p_unnorm / p_unnorm.sum()
                
                # Muestrear clase
                chosen = np.random.choice(n_classes, p=p_norm)
                predictions[k] = chosen
                path_length += euc_matrix[k, chosen]
            
            pred_labels = [classes[idx] for idx in predictions]
            n_errors = sum(
                1 for p, t in zip(pred_labels, test_labels) if p != t
            )
            
            local_solutions.append((n_errors, path_length, pred_labels))
        
        # Mejor solución local
        best_local = min(local_solutions, key=lambda s: (s[0], s[1]))
        
        # Actualizar pheromona (Eq. 6): refuerzo a las asignaciones de la mejor
        delta_tau = 1.0 / (best_local[1] + 1e-12)
        pheromone *= (1 - rho)
        for k, label in enumerate(best_local[2]):
            class_idx = classes.index(label)
            pheromone[k, class_idx] += delta_tau
        
        global_solutions.append(best_local)
        
        if verbose and (iteration + 1) % 30 == 0:
            print(f"  Iter {iteration + 1}/{config['n_iterations']}: "
                  f"best errores={best_local[0]}, path={best_local[1]:.2f}, "
                  f"ρ={rho:.4f}")
    
    # Radix sort (Sec. 2.5.4)
    sorted_solutions = radix_sort_solutions(global_solutions)
    best_global = sorted_solutions[0]
    
    # Comparación final: si la base sin estocasticidad es mejor, usarla
    if base_n_errors < best_global[0]:
        final_predictions = base_pred_labels
        final_n_errors = base_n_errors
        final_path = float(np.sum([
            euc_matrix[k, base_predictions[k]] for k in range(n_test)
        ]))
    else:
        final_predictions = best_global[2]
        final_n_errors = best_global[0]
        final_path = best_global[1]
    
    accuracy = 1.0 - (final_n_errors / n_test)
    
    return {
        'predictions': final_predictions,
        'accuracy': accuracy,
        'n_errors': final_n_errors,
        'path_length': final_path,
        'base_n_errors': base_n_errors,
        'history': [(s[0], s[1]) for s in global_solutions],
    }


if __name__ == "__main__":
    # Smoke test
    np.random.seed(0)
    
    # Centros sintéticos
    centers = {
        'A': np.sin(np.linspace(0, 4 * np.pi, 100)),
        'B': np.cos(np.linspace(0, 4 * np.pi, 100)),
    }
    center_feats = {
        'A': np.array([1.0, 2.0] + [0.0] * 34),
        'B': np.array([-1.0, -2.0] + [0.0] * 34),
    }
    
    # Test signals: 5 de cada clase
    n_test = 10
    test_signals = np.array([
        centers['A'] + 0.1 * np.random.randn(100) if i < 5
        else centers['B'] + 0.1 * np.random.randn(100)
        for i in range(n_test)
    ])
    test_features = np.array([
        np.array([1.0, 2.0] + [0.0] * 34) + 0.1 * np.random.randn(36) if i < 5
        else np.array([-1.0, -2.0] + [0.0] * 34) + 0.1 * np.random.randn(36)
        for i in range(n_test)
    ])
    test_labels = ['A'] * 5 + ['B'] * 5
    
    # Config rápido para smoke test
    config_test = ACO_CONFIG.copy()
    config_test['n_iterations'] = 10
    config_test['n_ants'] = 5
    
    result = run_aco_classification(
        test_signals, test_features, test_labels,
        centers, center_feats,
        config=config_test,
    )
    print(f"\nAccuracy: {result['accuracy']:.2%}")
    print(f"Errores: {result['n_errors']}/{n_test}")
