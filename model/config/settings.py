# Paramètres par défaut du système
DEFAULT_XTTS_CONFIG = {
    "supported_languages":
        ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"],
    "sample_rate": 22050,
    "model_channels": 1024,
    "emotion_dim": 256
}

# Classes de validation des paramètres
class TrainingConfig:
    def __init__(self, config_dict):
        self.batch_size = config_dict.get('batch_size', 4)
        self.learning_rate = config_dict.get('learning_rate', 1e-4)
        self.num_epochs = config_dict.get('num_epochs', 100)

class VADConfig:
    def __init__(self, config_dict):
        self.vad_eval_frequency = config_dict.get('vad_eval_frequency', 10)
        self.low_accuracy_threshold = config_dict.get('low_accuracy_threshold', 0.7)

# Fonctions de validation
def validate_config(config):
    # TODO
    pass
