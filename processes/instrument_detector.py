"""
Load a hf audio classification encoder (& decoder) and run inference on a WAV file
Automatically downloads the model if not present locally

Interface required by app.py:
NAME: str
DESCRIPTION: str (optional)
add_arguments(parser)
run(args)
"""

import os
import argparse
from pathlib import Path

NAME = "instrument_detector"
DESCRIPTION = "classify an audio file using an audio classification en- & decoder (or model if .yml available)"

DEFAULT_MODEL_NAME = "dima806/musical_instrument_detection"
MODEL_DIR = Path("models") / DEFAULT_MODEL_NAME.split("/")[-1]
DEFAULT_TEST_WAV = Path("test_wav") / "test.wav"

def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model-path", "-m", type=str,
        default=str(MODEL_DIR),
        help=f"Path to model directory. Default: '{MODEL_DIR}'"
    )
    parser.add_argument(
        "--from-hub", action="store_true",
        help="Force download the model from HF-Hub even if local path exists."
    )
    parser.add_argument(
        "--hub-name", type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model ID on HF. Default: {DEFAULT_MODEL_NAME}"
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HF requires an auth-token to download models from the hub. 'hf_somerandomnumbers' (or set HF_TOKEN in root/.env)"
    )
    parser.add_argument(
        "--test", nargs="?", const=str(DEFAULT_TEST_WAV), default=None,
        help=f"Quick test to check if everything works as intendet. You can optionally add '--test /path/to/file.wav' or simply replace {DEFAULT_TEST_WAV}"
    )
    parser.add_argument(
        "--file", "-f", type=str, default=None,
        help="'-f' or '--file /path/to/file.wav'. If omitted and no --test provided, the CLI look for 'test_wav/test.wav' and otherwise ask for input. <3 MewMew <3"
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate (default: 16000)")
    parser.add_argument("--max-length-seconds", type=int, default=15, help="Max audio length in seconds")
    parser.add_argument("--topk", type=int, default=5, help="Show top-k predictions")
    parser.add_argument("--live", action="store_true", help="Run live mode. (Currently) only a placeholder")

def _download_model_from_hub(model_path: Path, hub_name: str, token: str | None = None):
    """Download model from HF Hub if not already present"""
    if model_path.exists() and not any(model_path.iterdir()):
        model_path.rmdir()
    if model_path.is_dir():
        print(f"Using local model at: {model_path}")
        return

    print(f"Model not found at {model_path}. Downloading from HF-Hub: {hub_name}... Please wait! If download fails manually delete {model_path}")
    try:
        from huggingface_hub import snapshot_download

        os.makedirs(str(model_path), exist_ok=True)
        try:
            snapshot_download(repo_id=hub_name, local_dir=str(model_path), local_dir_use_symlinks=False, token=token)
        except TypeError:
            snapshot_download(repo_id=hub_name, local_dir=str(model_path), local_dir_use_symlinks=False, use_auth_token=token)
        print(f"Model downloaded and saved to {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download model '{hub_name}' from HF-Hub: {e}") from e

def run(args):
    # placeholder for live-input-feature
    if getattr(args, "live", False):
        print("Coming soon...")
        return

    try:    
        import torch
        import torchaudio
        import numpy as np
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        torchaudio.set_audio_backend("soundfile")
        
    except Exception as e:
        raise RuntimeError("Missing or invalid dependency. Run 'pip install -r requirements.txt'") from e

    model_path = Path(args.model_path).expanduser().resolve()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("AUTH_KEY")
    hub_name = args.hub_name
    
    file_arg = args.file
    test_arg = args.test
    resolved_file = None

    if test_arg is not None:
        candidate = Path(test_arg).expanduser()
        if not candidate.is_file():
            # user passed --test without value
            if test_arg == str(DEFAULT_TEST_WAV) and DEFAULT_TEST_WAV.is_file():
                resolved_file = DEFAULT_TEST_WAV.resolve()
            else:
                raise FileNotFoundError(f"--test was given but file not found: {candidate}")
        else:
            resolved_file = candidate.resolve()
    elif file_arg:
        candidate = Path(file_arg).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"Brrrrrrrrrr, input file is invalid or not found: {candidate}")
        resolved_file = candidate.resolve()
    else:
        # no explicit file args
        if DEFAULT_TEST_WAV.is_file():
            resolved_file = DEFAULT_TEST_WAV.resolve()
            print(f"Found {resolved_file}, using it.")
        else:
            # quality of life prompt
            user_path = input("Oida! Wenn du das hier liest, geht das nächste Bier auf dich! <3 Enter path to .wav file: ").strip()
            candidate = Path(user_path).expanduser()
            if not candidate.is_file():
                raise FileNotFoundError(f"Phew! Mehr quality-of-life geht echt nimmer. File not found: {candidate}")
            resolved_file = candidate.resolve()

    target_sr = args.sample_rate
    max_length = args.max_length_seconds * target_sr
    topk = max(1, args.topk)

    if args.from_hub or not model_path.is_dir():
        _download_model_from_hub(model_path, hub_name, hf_token)

    print(f"Loading feature extractor and model from: {model_path}")
    try: 
        feature_extractor = AutoFeatureExtractor.from_pretrained(str(model_path))
        model = AutoModelForAudioClassification.from_pretrained(str(model_path))
    except Exception as e:
        raise RuntimeError(f"Could not load model from {model_path}. Is it a valid HF format? Error: {e}") from e

    model.eval()
    id2label = getattr(model.config, "id2label", {}) or {}

    def preprocess_audio(file_path):
        waveform, sr = torchaudio.load(str(file_path))
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        # convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio = waveform.squeeze(0).numpy()

        if len(audio) > max_length:
            duration_original = len(audio) / target_sr
            print(f"Audio is {duration_original:.1f}s, truncating to {args.max_length_seconds}s")
            audio_proc = audio[:max_length]
        elif len(audio) < max_length:
            pad = np.zeros(max_length - len(audio), dtype=audio.dtype)
            audio_proc = np.concatenate([audio, pad])
        else:
            audio_proc = audio
        return audio_proc

    print("Preprocessing audio...")
    audio = preprocess_audio(resolved_file)

    print("Extracting features and running inference...")
    inputs = feature_extractor(audio, sampling_rate=target_sr, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

    results = [(id2label.get(i, f"Label_{i}"), float(score)) for i, score in enumerate(probs)]
    results.sort(key=lambda x: x[1], reverse=True)

    print("\nTop predictions:")
    for label, score in results[:topk]:
        print(f" - {label}: {score:.4f}")