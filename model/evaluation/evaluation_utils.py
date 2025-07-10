import numpy as np
from typing import List, Tuple, Any
from numpy import floating

def calculate_emotion_distance(
        pred_valence: float,
        pred_arousal: float,
        target_valence: float,
        target_arousal: float
) -> float:
    """Calculate Euclidean distance in emotion space."""
    return np.sqrt((pred_valence - target_valence)**2 + (pred_arousal - target_arousal)**2)

def emotion_quadrant_accuracy(
        pred_valence: float,
        pred_arousal: float,
        target_valence: float,
        target_arousal: float
) -> bool:
    """Check if prediction is in same emotion quadrant as target."""
    pred_quadrant = get_emotion_quadrant(pred_valence, pred_arousal)
    target_quadrant = get_emotion_quadrant(target_valence, target_arousal)
    return pred_quadrant == target_quadrant

def get_emotion_quadrant(valence: float, arousal: float) -> str:
    """Get emotion quadrant based on valence/arousal values."""
    v_center, a_center = 0.5, 0.5

    if valence >= v_center and arousal >= a_center:
        return "high_valence_high_arousal"  # Joie, excitation
    elif valence >= v_center and arousal < a_center:
        return "high_valence_low_arousal"   # Calme, satisfaction
    elif valence < v_center and arousal >= a_center:
        return "low_valence_high_arousal"   # Colère, peur
    else:
        return "low_valence_low_arousal"    # Tristesse, ennui

def batch_emotion_metrics(
        predictions: List[Tuple[float, float]],
        targets: List[Tuple[float, float]]
) -> dict[str, floating[Any] | Any]:
    """Calculate comprehensive emotion metrics for a batch."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    distances = []
    quadrant_matches = []
    valence_errors = []
    arousal_errors = []

    for (pred_v, pred_a), (targ_v, targ_a) in zip(predictions, targets):
        distances.append(calculate_emotion_distance(pred_v, pred_a, targ_v, targ_a))
        quadrant_matches.append(emotion_quadrant_accuracy(pred_v, pred_a, targ_v, targ_a))
        valence_errors.append(abs(pred_v - targ_v))
        arousal_errors.append(abs(pred_a - targ_a))

    return {
        'mean_emotion_distance': np.mean(distances),
        'quadrant_accuracy': np.mean(quadrant_matches),
        'valence_mae': np.mean(valence_errors),
        'arousal_mae': np.mean(arousal_errors),
        'valence_rmse': np.sqrt(np.mean(np.array(valence_errors)**2)),
        'arousal_rmse': np.sqrt(np.mean(np.array(arousal_errors)**2))
    }
