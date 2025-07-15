from audio_processor import ProcessorConfig, UniversalAudioProcessor

if __name__ == "__main__":
    config = ProcessorConfig(
        output_dir="processed_datasets",
        input_datasets={
            "emovdb": "../data_collection/tts_data/raw/emovdb",
            "iemocap": "../data_collection/tts_data/raw/iemocap/data",
            "cremad": "../data_collection/tts_data/raw/cremad",
            "ravdess": "../data_collection/tts_data/raw/ravdess"
        },
        whisper_model="small",
        target_sr=22050,
        min_duration=1.0,
        max_duration=30.0,
        min_transcript_length=3,
        verbose=True,
        progress_interval=100
    )
    
    processor = UniversalAudioProcessor(config)
    
    for dataset_name, dataset_path in config.input_datasets.items():
        print(f"\nProcessing {dataset_name} from {dataset_path}...")
        
        # Test with 5 files first if you want
        results = processor.process_dataset(
            input_dir=dataset_path,
            dataset_name=f"{dataset_name}_test",
            max_files=5,
            random_sample=False
        )
        
        # Full processing
        #results = processor.process_dataset(
        #    input_dir=dataset_path,
        #    dataset_name=dataset_name
        #)
        
        print(f"Completed {dataset_name}: {results}")
    
    print(f"\nAll processing complete! Check {config.output_dir} for results.")