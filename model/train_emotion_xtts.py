from model.training.trainer import EmotionalXTTSTrainer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train VAD-Guided Emotional XTTS model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    trainer = EmotionalXTTSTrainer(args.config)
    
    if args.resume:
        print(f"Resuming training from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    trainer.train()


if __name__ == "__main__":
    main()