"""One-time download. The running application uses local files only."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
REVISION = "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"

if __name__ == "__main__":
    target = ROOT / "models" / "base"
    snapshot_download("Systran/faster-whisper-base", revision=REVISION,
                      local_dir=target,
                      allow_patterns=["config.json", "model.bin", "tokenizer.json",
                                      "vocabulary.*", "preprocessor_config.json"])
    (target / "download_identity.json").write_text(json.dumps({
        "repository": "Systran/faster-whisper-base", "revision": REVISION
    }, indent=2), encoding="utf-8")
    print("Model downloaded. Local transcription can now run without internet.")
