import torch
import numpy as np
from typing import Dict


def get_vad_guided_targets(model, target_valence, target_arousal):
    emotion_magnitude = torch.sqrt(target_valence**2 + target_arousal**2)

    target_gpt_modification = torch.tensor(
        model.adaptive_gpt_strength * (0.5 + emotion_magnitude.item()),
        requires_grad=False,
        device=target_valence.device
    )

    target_speaker_modification = torch.tensor(
        model.adaptive_speaker_strength * (0.5 + emotion_magnitude.item()),
        requires_grad=False,
        device=target_valence.device
    )

    return target_gpt_modification, target_speaker_modification

def update_vad_guided_targets(model, vad_accuracy):
    model.vad_feedback_history.append(vad_accuracy)
    if len(model.vad_feedback_history) > model.max_feedback_history:
        model.vad_feedback_history.pop(0)

    recent_accuracy = sum(model.vad_feedback_history[-model.recent_history_window:]) / min(model.recent_history_window, len(model.vad_feedback_history))

    if recent_accuracy < model.low_accuracy_threshold:  # VAD shows we're not hitting targets
        model.adaptive_gpt_strength *= model.increase_rate_gpt
        model.adaptive_speaker_strength *= model.increase_rate_speaker
        print(f"Increasing conditioning strength: GPT={model.adaptive_gpt_strength:.3f} (accuracy: {recent_accuracy:.3f})")
    elif recent_accuracy > model.high_accuracy_threshold:  # VAD shows we're very accurate
        model.adaptive_gpt_strength *= model.decrease_rate_gpt
        model.adaptive_speaker_strength *= model.decrease_rate_speaker
        print(f"Decreasing conditioning strength: GPT={model.adaptive_gpt_strength:.3f} (accuracy: {recent_accuracy:.3f})")

    model.adaptive_gpt_strength = max(model.min_gpt_strength, min(model.max_gpt_strength, model.adaptive_gpt_strength))
    model.adaptive_speaker_strength = max(model.min_speaker_strength, min(model.max_speaker_strength, model.adaptive_speaker_strength))


class EmotionMetrics:
    """Classe pour calculer les métriques émotionnelles."""

    def __init__(self):
        self.arousal_targets = None
        self.arousal_predictions = None
        self.valence_targets = None
        self.valence_predictions = None
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self.valence_predictions = []
        self.valence_targets = []
        self.arousal_predictions = []
        self.arousal_targets = []

    def update(self, pred_valence: float, pred_arousal: float,
               target_valence: float, target_arousal: float):
        """Update metrics with new predictions."""
        self.valence_predictions.append(pred_valence)
        self.valence_targets.append(target_valence)
        self.arousal_predictions.append(pred_arousal)
        self.arousal_targets.append(target_arousal)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        if not self.valence_predictions:
            return {'valence_mae': 0.0, 'arousal_mae': 0.0, 'emotion_accuracy': 0.0}

        valence_mae = np.mean(np.abs(np.array(self.valence_predictions) - np.array(self.valence_targets)))
        arousal_mae = np.mean(np.abs(np.array(self.arousal_predictions) - np.array(self.arousal_targets)))

        # Calcul de la précision émotionnelle globale
        emotion_accuracy = max(0.0, 1.0 - (valence_mae + arousal_mae) / 2.0)

        return {
            'valence_mae': valence_mae,
            'arousal_mae': arousal_mae,
            'emotion_accuracy': emotion_accuracy,
            'valence_rmse': np.sqrt(np.mean((np.array(self.valence_predictions) - np.array(self.valence_targets))**2)),
            'arousal_rmse': np.sqrt(np.mean((np.array(self.arousal_predictions) - np.array(self.arousal_targets))**2))
        }
