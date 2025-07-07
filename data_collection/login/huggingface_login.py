from huggingface_hub import login, HfFolder
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv('HUGGINGFACE_TOKEN')

if hf_token:
    try:
        login(token=hf_token)
        print("✓ Successfully logged in to Hugging Face!")
        
        # Verify login worked
        whoami = HfFolder.get_token()
        if whoami:
            print("✓ Authentication verified")
        else:
            print("❌ Login failed - token may be invalid")
            
    except Exception as e:
        print(f"❌ Login failed: {e}")
else:
    print("❌ HUGGINGFACE_TOKEN not found in .env file")
    print("Please add: HUGGINGFACE_TOKEN=your_token_here to your .env file")