DATASETS = {
    "iemocap": {
        "downloader": "huggingface",
        "name": "AbstractTTS/IEMOCAP",
        "description": "Interactive Emotional Dyadic Motion Capture Database"
    },
    "cremad": {
        "downloader": "kaggle", 
        "name": "ejlok1/cremad",
        "description": "Crowdsourced Emotional Multimodal Actors Dataset"
    },
    "ravdess": {
        "downloader": "kaggle",
        "name": "uwrfkaggler/ravdess-emotional-speech-audio", 
        "description": "Ryerson Audio-Visual Database of Emotional Speech and Song"
    },
    "emovdb": {
        "downloader": "openslr",
        "name": "EmoV-DB",
        "description": "Emotional Voices Database",
        "openslr_id": 115,
        "files": [
            "bea_Amused.tar.gz", "bea_Angry.tar.gz", "bea_Disgusted.tar.gz",
            "bea_Neutral.tar.gz", "bea_Sleepy.tar.gz", "jenie_Amused.tar.gz",
            "jenie_Angry.tar.gz", "jenie_Disgusted.tar.gz", "jenie_Neutral.tar.gz",
            "jenie_Sleepy.tar.gz", "josh_Amused.tar.gz", "josh_Neutral.tar.gz",
            "josh_Sleepy.tar.gz", "sam_Amused.tar.gz", "sam_Angry.tar.gz",
            "sam_Disgusted.tar.gz", "sam_Neutral.tar.gz", "sam_Sleepy.tar.gz"
        ]
    },
    #"msimprov": {
    #    "downloader": "kaggle",
    #    "name": "pratyanchasharma01/ms-improv",
    #    "description": "MSP-IMPROV: Fraction of Audio-Visual Dataset from Original MS-IMPROV Corpus"
    #}
}