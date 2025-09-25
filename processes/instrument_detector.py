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
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
import time
import threading
import queue
from . import osc_manager

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
    parser.add_argument("--live", action="store_true", help="Run live mode with GUI and microphone")
    parser.add_argument("--chunk", type=float, default=1.0, help="Audio chunk duration in seconds for live mode")

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

class LiveAudioClassifier:
    def __init__(self, model, feature_extractor, id2label, target_sr=16000, max_length=None):
        self.model = model
        self.feature_extractor = feature_extractor
        self.id2label = id2label
        self.target_sr = target_sr
        self.max_length = max_length or (15 * target_sr)
        self.audio_queue = queue.Queue()
        self.prediction_queue = queue.Queue()
        self.recording = False

    def preprocess_audio(self, audio_data):
        start_time = time.time()

        import torch
        print(f"PyTorch OpenMP: {torch.get_num_threads()} threads")
        import numpy as np
        print(f"NumPy BLAS info: {np.show_config()}")

        # converting to torch tensor (if needed)
        if isinstance(audio_data, np.ndarray):
            waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
        else:
            waveform = audio_data

        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        elif waveform.shape[1] < self.max_length:
            pad_length = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        audio = waveform.squeeze(0).numpy()

        preprocess_time = time.time() - start_time
        return audio, preprocess_time

    def run_inference(self, audio):
        import torch

        start_time = time.time()

        inputs = self.feature_extractor(audio, sampling_rate=self.target_sr, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

        inference_time = time.time() - start_time

        # get top 3 results
        results = [(self.id2label.get(i, f"Label_{i}"), float(score)) for i, score in enumerate(probs)]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:3], inference_time

    def audio_capture_worker(self, chunk_duration=1.0):
        try:
            import pyaudio
            import numpy as np
        except ImportError:
            raise RuntimeError("'pyaudio' required for live mode. Install it with 'pip install pyaudio'")

        chunk_samples = int(self.target_sr * chunk_duration)

        p = pyaudio.PyAudio()

        # show mic --> microphone_manager.py
        default_device = p.get_default_input_device_info()
        print(f"Using microphone: {default_device['name']} (ID: {default_device['index']})")
        print(f"Sample rate: {int(default_device['defaultSampleRate'])} Hz")

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.target_sr,
            input=True,
            frames_per_buffer=1024
        )

        print(f"Recording audio in {chunk_duration}s chunks...")

        try:
            while self.recording:
                frames=[]
                for _ in range(0, int(self.target_sr / 1024 * chunk_duration)):
                    if not self.recording:
                        break
                    data = stream.read(1024)
                    frames.append(np.frombuffer(data, dtype=np.float32))
                
                if frames and self.recording:
                    audio_chunk = np.concatenate(frames)
                    self.audio_queue.put(audio_chunk)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def inference_worker(self):
        while self.recording or not self.audio_queue.empty():
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                # measuring total processing time
                total_start = time.time()
                # preprocessing
                processed_audio, preprocess_time = self.preprocess_audio(audio_chunk)
                #running inference
                predictions, inference_time = self.run_inference(processed_audio)

                total_time = time.time() - total_start

                # queueing results for GUI
                timing_info = {
                    'preprocess_time': preprocess_time,
                    'inference_time': inference_time,
                    'total_time': total_time
                }

                self.prediction_queue.put((predictions, timing_info))

                if hasattr(self, 'osc_enabled') and self.osc_enabled:
                    osc_manager.update_osc_predictions(predictions)

            except queue.Empty:
                continue

    def start_live_classification(self, chunk_duration=1.0):
        self.recording = True

        # starting audio thread
        audio_thread = threading.Thread(target=self.audio_capture_worker, args=(chunk_duration,))
        audio_thread.daemon = True
        audio_thread.start()

        # starting audio capture thread
        inference_thread = threading.Thread(target=self.inference_worker)
        inference_thread.daemon = True
        inference_thread.start()

        return audio_thread, inference_thread

    def stop_live_classification(self):
        self.recording = False

def create_live_gui(classifier, chunk_duration=1.0):
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        raise RuntimeError("Brrrrrrrrr, tkinter is a critical requirement! Something is seriously wrong, because it should be included in Python...")

    root = tk.Tk()
    root.title("Live Instrument Detection")
    root.geometry("500x400")
    root.configure(bg='#2b2b2b')

    title_label = tk.Label(root, text="<3 MewMew <3", font=('Arial', 16, 'bold'), bg='#2b2b2b', fg='white')
    title_label.pack(pady=10)

    status_var = tk.StringVar(value="Ready to start...")
    status_label = tk.Label(root, textvariable=status_var, font=('Arial', 12), bg='#2b2b2b', fg='#00ff00')
    status_label.pack(pady=5)

    # predictions
    pred_frame = tk.Frame(root, bg='#2b2b2b')
    pred_frame.pack(pady=20, padx=20, fill='both', expand=True)

    pred_labels = []
    for i in range(3):
        frame = tk.Frame(pred_frame, bg='#3b3b3b', relief='raised', bd=2)
        frame.pack(fill='x', pady=5)

        rank_label = tk.Label(frame, text=f"#{i+1}", font=('Arial', 14, 'bold'), bg='#3b3b3b', fg='#ffaa00')
        rank_label.pack(side='left', padx=10, pady=10)
        
        name_label = tk.Label(frame, text="---", font=('Arial', 12, 'bold'), bg='#3b3b3b', fg='white')
        name_label.pack(side='left', padx=10, pady=10)

        score_label = tk.Label(frame, text="0.00%", font=('Arial', 12, 'bold'), bg='#3b3b3b', fg='#00ffff')
        score_label.pack(side='right', padx=10, pady=10)

        pred_labels.append((name_label, score_label))

    timing_var = tk.StringVar(value="Timing: --- ms")
    timing_label = tk.Label(root, textvariable=timing_var, font=('Arial', 10), bg='#2b2b2b', fg='#888888')
    timing_label.pack(pady=5)

    osc_frame = tk.Frame(root, bg='#2b2b2b')
    osc_frame.pack(pady=5, padx=20, fill='x')

    osc_status_var = tk.StringVar(value="OSC: localhost:9000")
    osc_status_label = tk.Label(osc_frame, textvariable=osc_status_var, font=('Arial', 10, 'bold'), bg='#2b2b2b', fg='#00ff00')
    osc_status_label.pack(side='left')

    osc_counter_var = tk.StringVar(value="Sent: 0")
    osc_counter_label = tk.Label(osc_frame, textvariable=osc_counter_var, font=('Arial', 10), bg='#2b2b2b', fg='#ffaa00')
    osc_counter_label.pack(side='right')

    button_frame = tk.Frame(root, bg='#2b2b2b')
    button_frame.pack(pady=10)

    recording = False

    def toggle_recording():
        nonlocal recording
        if not recording:
            recording = True
            start_btn.config(text="STOP", bg='#ff4444')
            status_var.set("Recording...")
            classifier.start_live_classification(chunk_duration)
        else:
            recording = False
            start_btn.config(text="Start", bg='#44ff44')
            status_var.set("Stopped")
            classifier.stop_live_classification()

    start_btn = tk.Button(button_frame, text="START", command=toggle_recording, font=('Arial', 12, 'bold'), bg='#44ff44', fg='black', padx=20, pady=10)
    start_btn.pack(side='left', padx=10)

    quit_btn = tk.Button(button_frame, text="Quit", command=root.quit, font=('Arial', 12, 'bold'), bg='#ff4444', fg='white', padx=20, pady=10)
    quit_btn.pack(side='left', padx=10)

    def update_predictions():
        try:
            predictions, timing_info = classifier.prediction_queue.get_nowait()
            for i, (name_label, score_label) in enumerate(pred_labels):
                if i < len(predictions):
                    label, score = predictions[i]
                    name_label.config(text=label)
                    score_label.config(text=f"{score:.1%}")
                else:
                    name_label.config(text="---")
                    score_label.config(text="0.00%")

            total_ms = timing_info['total_time'] * 1000
            inference_ms = timing_info['inference_time'] * 1000
            timing_var.set(f"Inference: {inference_ms:.0f}ms | Total: {total_ms:.0f}ms")

            if hasattr(classifier, 'osc_enabled') and classifier.osc_enabled:
                osc_info = osc_manager.get_osc_status()
                if osc_info:
                    osc_status_var.set(f"OSC: {osc_info['host']}:{osc_info['port']}")
                    osc_counter_var.set(f"Sent: {osc_info['message_count']} | Last: {osc_info['last_send']}")
                else:
                    osc_status_var.set("OSC: ERROR")
            else:
                osc_status_var.set("OSC: Disabled")

        except queue.Empty:
            pass

        root.after(100, update_predictions)
        
    #schedule next update
    root.after(100, update_predictions)

    return root

def run(args):
    # live mode
    if getattr(args, "live", False):
        try:
            import torch
            import torchaudio
            import numpy as np
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            torchaudio.set_audio_backend("soundfile")
        except Exception as e:
            raise RuntimeError("Brrrrrrr, missing or invalid dependency, Run 'pip install -r requirements.txt'") from e

        model_path = Path(args.model_path).expanduser().resolve()
        hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("AUTH_KEY")
        hub_name = args.hub_name

        if args.from_hub or not model_path.is_dir():
            _download_model_from_hub(model_path, hub_name, hf_token)

        print(f"Loading model...")
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(str(model_path))
            model = AutoModelForAudioClassification.from_pretrained(str(model_path))
        except Exception as e:
            raise RuntimeError(f"Could not load model from {model_path}. Error: {e}") from e

        model.eval()
        id2label = getattr(model.config, "id2label", {}) or {}
        classifier = LiveAudioClassifier(model, feature_extractor, id2label, args.sample_rate)

        osc_manager.start_osc_server(args.chunk)
        classifier.osc_enabled = True
        print(f"OSC server started - spamming localhost:9000 every {args.chunk}s for 'bussi aufs bauchi'")

        gui = create_live_gui(classifier, args.chunk)

        print("Starting live mode GUI for se Schmiede moopsies...")
        gui.mainloop()

        #Cleanup
        classifier.stop_live_classification()
        osc_manager.stop_osc_server()
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