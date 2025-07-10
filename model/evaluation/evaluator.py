import torch
from typing import Dict, Any

from tqdm import tqdm

from model.training.loss_functions import compute_conditioning_loss
from model.utils.model_utils import save_temp_audio, cleanup_temp_file, generate_audio_sample


class EmotionEvaluator:
    """Dedicated class for evaluating emotion-based audio generation models."""

    def __init__(self, model, device):
        self.val_dataloader = None
        self.model = model
        self.device = device

    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a validation step on a batch."""
        with torch.no_grad():
            try:
                batch_size = len(batch['texts'])
                total_conditioning_loss = 0.0
                valid_samples = 0

                vad_metrics = {
                    'valence_mae': 0.0,
                    'arousal_mae': 0.0
                }

                for i in range(batch_size):
                    try:
                        target_valence = batch['target_valences'][i].to(self.device)
                        target_arousal = batch['target_arousals'][i].to(self.device)

                        # Compute conditioning loss
                        conditioning_loss, cond_info = compute_conditioning_loss(
                            self.model, batch['speaker_refs'][i], target_valence, target_arousal
                        )

                        # Generate and evaluate audio
                        generated_audio = generate_audio_sample(
                            self.model, batch['texts'][i], batch['speaker_refs'][i],
                            target_valence, target_arousal
                        )

                        # VAD analysis
                        vad_result = self.compute_vad_loss_for_validation(
                            generated_audio, target_valence, target_arousal
                        )

                        if torch.isfinite(conditioning_loss) and vad_result:
                            total_conditioning_loss += conditioning_loss.item()
                            valid_samples += 1
                            vad_metrics['valence_mae'] += abs(vad_result['pred_valence'] - target_valence.item())
                            vad_metrics['arousal_mae'] += abs(vad_result['pred_arousal'] - target_arousal.item())

                    except Exception as e:
                        print(f"Error in validation sample {i}: {e}")
                        continue

                if valid_samples > 0:
                    return {
                        'conditioning_loss': total_conditioning_loss / valid_samples,
                        'valence_mae': vad_metrics['valence_mae'] / valid_samples,
                        'arousal_mae': vad_metrics['arousal_mae'] / valid_samples,
                        'valid_samples': valid_samples
                    }
                else:
                    return {
                        'conditioning_loss': 0.0,
                        'valence_mae': 0.0,
                        'arousal_mae': 0.0,
                        'valid_samples': 0
                    }

            except Exception as e:
                print(f"Error in validation step: {e}")
                return {
                    'conditioning_loss': 0.0,
                    'valence_mae': 0.0,
                    'arousal_mae': 0.0,
                    'valid_samples': 0
                }

    def validate_epoch(self, epoch):
        self.model.model.eval()
        epoch_metrics = {
            'conditioning_loss': 0.0,
            'valence_mae': 0.0,
            'arousal_mae': 0.0,
            'valid_samples': 0
        }
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                metrics = self.validation_step(batch)

                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
                num_batches += 1

        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches

        self.model.writer.add_scalar('Val/ConditioningLoss', epoch_metrics['conditioning_loss'], epoch)
        self.model.writer.add_scalar('Val/ValenceMAE', epoch_metrics['valence_mae'], epoch)
        self.model.writer.add_scalar('Val/ArousalMAE', epoch_metrics['arousal_mae'], epoch)

        return epoch_metrics

    def compute_vad_loss_for_validation(
            self,
            generated_audio: torch.Tensor,
            target_valence: torch.Tensor,
            target_arousal: torch.Tensor
    ) -> dict[str, int | float | bool | Any] | None:
        """Compute VAD loss for validation."""
        temp_path = None
        try:
            generated_audio_cpu = generated_audio.detach().cpu()
            temp_path = save_temp_audio(generated_audio_cpu)
            vad_result, status = self.model.vad_analyzer.extract(temp_path)

            if status != "success" or vad_result is None:
                print(f"VAD analysis failed: {status}")
                return None

            return {
                'pred_valence': vad_result['valence'],
                'pred_arousal': vad_result['arousal'],
                'target_valence': target_valence.item(),
                'target_arousal': target_arousal.item()
            }

        except Exception as e:
            print(f"Error computing VAD loss: {e}")
            return None

        finally:
            if temp_path:
                cleanup_temp_file(temp_path)
