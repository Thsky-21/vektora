"""
deploy.py — publish the demo to Hugging Face Spaces.

One-time setup, in YOUR OWN terminal (it asks for your token; never paste the
token into code, chat, or GitHub):
    .venv/Scripts/hf.exe auth login

Then, from the repo root:
    .venv/Scripts/python.exe demo/deploy.py

It creates the Space if it doesn't exist yet (Docker SDK, public, free CPU)
and uploads only what the website needs. It does NOT upload demo/data/
(your raw recordings, including your phone talking).
"""

import os

from huggingface_hub import HfApi

SPACE = "ThSky21/vektora-demo"
HERE = os.path.dirname(os.path.abspath(__file__))

# Only these files go online.
UPLOAD = [
    "app.py", "audio_utils.py", "plots.py", "prepare_data.py", "train.py",
    "requirements.txt", "Dockerfile", "README.md",
    "model/*",            # the trained network + normalisation + threshold
    "plots/*.png",        # training graphs shown in the app
    "examples/*.wav",     # held-out test clips for the "Example clips" button
]

if __name__ == "__main__":
    for needed in ("model/vektora.keras", "model/norm.json", "model/metrics.json"):
        if not os.path.exists(os.path.join(HERE, needed)):
            raise SystemExit(f"missing {needed}: run demo/train.py first")
    api = HfApi()
    print(f"logged in as: {api.whoami()['name']}")
    api.create_repo(SPACE, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=HERE, repo_id=SPACE, repo_type="space",
                      allow_patterns=UPLOAD, commit_message="Update Vektora demo")
    print(f"\nuploaded. Build log + app: https://huggingface.co/spaces/{SPACE}")
    print("First build takes ~5-10 min (TensorFlow is large). Watch the 'Logs' tab.")
