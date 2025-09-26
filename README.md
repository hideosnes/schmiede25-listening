# Real-time Audio Classification

**Version:** v0.0.3 (alpha release)  
**Contributions:** Not unwelcome! Please mark AI-generated code as such in your commits.

This application provides real-time audio classification using PyTorch-based transformer models from Hugging Face. The system captures live microphone input, processes it in configurable chunks (default: 1-second intervals), and runs inference to identify instruments or sounds. Results are displayed via a Tkinter GUI and automatically broadcast via OSC (Open Sound Control) to `localhost:9000` for integration with DAWs and music software.

The architecture supports:
- File-based analysis and continuous live processing
- Automatic model downloading from Hugging Face Hub
- Multiple microphone device selection
- Synchronized OSC messaging matching audio processing intervals

Default model recognizes over 500 audio classes from the AudioSet dataset, but users can specify alternative models for different tasks.

---

## Requirements

- Python 3.10+
- macOS or Linux (Windows untested)
- Microphone access permissions
- 4GB+ RAM (for larger models)
- Network access (for model downloads)

---

## Troubleshooting

### Environment Setup (macOS Recommended)

On macOS, we **strongly recommend** using `conda` and installing `portaudio` via Homebrew to avoid compilation issues with `scipy` and `numpy`. Apple’s default toolchain often causes problems ("your version of clang is not supported").

**Recommended**: Use [Miniforge](https://github.com/conda-forge/miniforge) (Conda + conda-forge by default).

---

### Audio Issues

**Linux:**
`sudo apt install portaudio19-dev python3-dev python3-tk`

**macOS:**
`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

`xcode-select --install`

Also ensure microphone permissions are enabled in System Settings.

> ALSA warnings on Linux are harmless and can be ignored.

---

### Model Download Issues

- Set Hugging Face token:  
  export HF_TOKEN=your_token_here
- Force re-download: use the `--from-hub` flag
- Check internet connection and firewall settings

---

## TUTORIAL

### Quick Start (macOS)

`conda activate schmiede25`

`python app.py instrument_detector --live`

### Quick Start (Other Systems)

`source venv/bin/activate` # Or your preferred virtual env manager

`python app.py instrument_detector --live`

---

## OSC Output

OSC is **automatically enabled in live mode**, broadcasting to `localhost:9000`.

### Addresses & Schema
Sends three messages per chunk (top-3 predictions):

- /instrument/top1 -> [str(label), float(score_clamped_0_to_1)]
- /instrument/top2 -> [str(label), float(score_clamped_0_to_1)]
- /instrument/top3 -> [str(label), float(score_clamped_0_to_1)]

> Note: OSC sends raw ASCII strings — decode appropriately in your receiving app.

---

## Manual Installation (Linux / Windows)
### Step 1:
`bash scripts/setup.sh`

### On Linux (if needed)
`sudo apt-get install portaudio19-dev`

`pip install pyaudio`

## Manual Installtion (macOS)
### Step 1: Install Miniforge (Conda-Forge)

[Install Mini-Forge](https://github.com/conda-forge/miniforge)

### Step 2: Create Conda Environment

`conda create -n myCondaName python=3.10`

`conda activate myCondaName`

### Step 3: Install Scientific Packages (via Conda)

`conda install scipy librosa numpy`

### Step 4: Install Python Packages

`pip install torch torchaudio transformers huggingface-hub soundfile`

### Step 5: Install pyaudio (Recommended via Homebrew)

# Install Homebrew first:
`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

# Then install portaudio and pyaudio
`brew install portaudio`

`pip install pyaudio`

### Step 6: Install Project Dependencies

`pip install -r requirements.txt`

> If you encounter issues installing `torch` on macOS, refer to [PyTorch official docs](https://pytorch.org). Contacting Apple support would be an alternative option. 😅

---

## Clone and Run

`git clone https://github.com/hideosnes/schmiede25-listening.git`

`cd schmiede25-listening`

`python app.py --list`

---

## ARGUMENTS

### Test Available Microphones

python app.py microphone_manager [OPTIONS]

**Options:**
- --list: List all available microphones/input devices
- --test ID: Test microphone by device ID
- --duration SECONDS: Duration of test (default: 5)

#### Examples

List devices:
python app.py microphone_manager --list

Test device ID 0 for 10 seconds:
python app.py microphone_manager --test 0 --duration 10

---

### Detect Instruments

python app.py instrument_detector [OPTIONS]

**Options:**
| Flag | Description | Default |
|------|-------------|--------|
| --model-path, -m PATH | Path to local model | models/musical_instrument_detection |
| --from-hub | Force download from HuggingFace | False |
| --hub-name NAME | Model ID on HuggingFace | dima806/musical_instrument_detection |
| --hf-token TOKEN | HuggingFace auth token | None |
| --test [FILE] | Quick test mode | test_wav/test.wav |
| --file, -f FILE | Audio file to analyze | — |
| --sample-rate RATE | Target sample rate | 16000 |
| --max-length-seconds SEC | Max audio length | 15 |
| --topk K | Show top-k predictions | 5 |
| --live | Live mode with GUI & mic input | False |
| --chunk DURATION | Chunk duration (seconds) | 1.0 |

---

### Usage Examples

Analyze a file:
python app.py instrument_detector --file path/to/audio.wav

Live mode with GUI and OSC:
python app.py instrument_detector --live

More responsive live mode (0.5s chunks):
python app.py instrument_detector --live --chunk 0.5

Quick test with default file:
python app.py instrument_detector --test

Show top 10 predictions:
python app.py instrument_detector --file audio.wav --topk 10

Force fresh model download:
python app.py instrument_detector --from-hub --live

---

## MODEL-ZOO

Pre-trained models compatible with this pipeline:

| Model | Description |
|-------|-------------|
| MIT/ast-finetuned-audioset-10-10-0.4593 | Recognizes 527+ audio classes (AudioSet) |
| facebook/wav2vec2-large-xlsr-53-german | Speech-focused, multilingual |
| facebook/hubert-large-ls960-ft | Good for wooden/brass instruments |
| microsoft/unispeech-sat-base-plus | Inconsistent performance |
| audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim | Emotion detection; accurate but slow |

### Example Usage

python app.py instrument_detector \
  --hub-name "MIT/ast-finetuned-audioset-10-10-0.4593" \
  --live

---

## ROADMAP & TODOs

### RAVE Integration
[RAVE](https://iil.is/news/ravemodels)  
Check out the Hugging Face repo: [huggingface.co/Intelligent-Instruments-Lab/rave-models](https://huggingface.co/Intelligent-Instruments-Lab/rave-models)  

RAVE (Realtime Audio Variational autoEncoder) is a neural synthesis model developed by Antoine Caillon and ACIDS/IRCAM. It learns:
- An **encoder** that extracts compressed latent features from audio
- A **decoder** that reconstructs high-quality sound from those features

Use cases:
- Decoder as standalone synthesizer
- Transform input audio to resemble training data (timbre transfer, voice conversion)
- Latent-space feature extraction
- Real-time sonification pipelines
- Transfer learning acceleration using provided checkpoints

### MIDI Generation
[Hugging Face: SAO-Instrumental-Finetune](https://huggingface.co/santifiorino/SAO-Instrumental-Finetune)  
Goal: Implement a Mixture-of-Experts (MoE) subset to generate structured MIDI data for controlling hardware/software synths.

### Model Cascade
Build a hierarchical classification pipeline:
1. **Binary**: Is this an instrument?
2. **Family**: Strings / Brass / Percussion / Keys / etc.
3. **Specific**: Violin, Trumpet, Snare Drum, etc.

Improves accuracy and reduces false positives.

### Plugin Support
Target external integrations:
- Pure Data
- Isadora

### NM-library
Develop a modular library for neural music applications, reusable across projects.
