import torch
import torch.nn.functional as F
from model.training.metrics import get_vad_guided_targets


def compute_conditioning_loss(model, speaker_ref_path, target_valence, target_arousal):
    try:
        device = model.device

        target_valence = target_valence.to(device)
        target_arousal = target_arousal.to(device)

        with torch.no_grad():
            original_gpt_latent, original_speaker_emb = model.model.xtts.get_conditioning_latents([speaker_ref_path])
            original_gpt_latent = original_gpt_latent.to(device)
            original_speaker_emb = original_speaker_emb.to(device)

        emotion_gpt_latent, emotion_speaker_emb = model.model.get_conditioning_latents_with_valence_arousal(
            speaker_ref_path, target_valence, target_arousal, training=True
        )

        emotion_gpt_latent = emotion_gpt_latent.to(device)
        emotion_speaker_emb = emotion_speaker_emb.to(device)

        gpt_diff = torch.norm(emotion_gpt_latent - original_gpt_latent)
        speaker_diff = torch.norm(emotion_speaker_emb - original_speaker_emb)

        target_gpt_modification, target_speaker_modification = get_vad_guided_targets(
            model,
            target_valence, target_arousal
        )

        gpt_loss = F.smooth_l1_loss(gpt_diff, target_gpt_modification)
        speaker_loss = F.smooth_l1_loss(speaker_diff, target_speaker_modification)

        reg_loss = 0.01 * (torch.norm(emotion_gpt_latent) + torch.norm(emotion_speaker_emb))

        total_loss = gpt_loss + speaker_loss + reg_loss

        return total_loss, {
            'gpt_diff': gpt_diff.item(),
            'speaker_diff': speaker_diff.item(),
            'target_gpt_mod': target_gpt_modification.item(),
            'target_speaker_mod': target_speaker_modification.item(),
            'gpt_loss': gpt_loss.item(),
            'speaker_loss': speaker_loss.item(),
            'reg_loss': reg_loss.item()
        }

    except Exception as e:
        print(f"Error computing conditioning loss: {e}")
        print(f"Device being used: {device}")
        print(f"Original GPT device: {original_gpt_latent.device if 'original_gpt_latent' in locals() else 'undefined'}")
        print(f"Emotion GPT device: {emotion_gpt_latent.device if 'emotion_gpt_latent' in locals() else 'undefined'}")
        print(f"Target valence device: {target_valence.device}")
        print(f"Target arousal device: {target_arousal.device}")
        raise e

def compute_speaker_similarity_loss(model, speaker_ref_path, generated_audio):
    temp_path = None
    try:
        device = model.device

        ref_gpt_latent, ref_speaker_emb = model.model.xtts.get_conditioning_latents([speaker_ref_path])
        ref_speaker_emb = ref_speaker_emb.to(device)

        generated_audio_cpu = generated_audio.detach().cpu()

        temp_path = model.save_temp_audio(generated_audio_cpu)
        gen_gpt_latent, gen_speaker_emb = model.model.xtts.get_conditioning_latents([temp_path])
        gen_speaker_emb = gen_speaker_emb.to(device)

        ref_speaker_emb = ref_speaker_emb.flatten()
        gen_speaker_emb = gen_speaker_emb.flatten()

        similarity = F.cosine_similarity(ref_speaker_emb, gen_speaker_emb, dim=0)

        speaker_loss = 1.0 - similarity

        return speaker_loss, similarity.item()

    except Exception as e:
        print(f"Error computing speaker similarity: {e}")
        return torch.tensor(0.5, requires_grad=True, device=model.device), 0.0

    finally:
        if temp_path:
            model.cleanup_temp_file(temp_path)