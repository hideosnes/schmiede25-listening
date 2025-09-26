# Real-time Audio Classification
Currently: v0.0.3 (alpha release)
Contributions are not unwelcome. Please mark AI generated code as such when you commit.

This application provides real-time audio classification using PyTorch-based transformer models from Hugging Face. The system captures live microphone input, processes it in configurable chunks (default 1-second intervals), and runs inference using pre-trained audio classification models to identify instruments or sounds. Results are displayed through a Tkinter GUI and automatically broadcast via OSC (Open Sound Control) protocol to localhost:9000 for integration with digital audio workstations and music software. The architecture supports both file-based analysis and continuous live processing, with automatic model downloading from Hugging Face Hub, multiple microphone device selection, and synchronized OSC messaging that matches the audio processing intervals. The default model recognizes over 500 audio classes from the AudioSet dataset, though users can specify alternative models for different classification tasks.

## Requirements
- Python 3.10+
- macOS/Linux (Windows untested)
- Microphone access permissions
- 4GB+ RAM for larger models
- Network access for model downloads

## Troubleshooting
On macOSX, it is strongly recommended to use 'conda' and to install 'oblast' to avoid issues with 'scipy' and 'numpy'. (Conda will provide binaries and handles building them in case!) We tested it successfully using Miniforge[https://github.com/conda-forge/miniforge]. Otherwise you'll encounter problems "with your version of clang" áka 'GTFO! Apple simply being Apple'.

**Audio Issues:**
- Linux: `sudo apt install portaudio19-dev python3-dev python3-tk`
- macOS: Install Xcode Command Line Tools (`xcode-select --install`)
- Check microphone permissions in system settings
- ALSA warnings on Linux are harmless

**Model Download Issues:**
- Set HuggingFace token: `export HF_TOKEN=your_token_here`
- Force re-download: `--from-hub` flag
- Check internet connection and firewall

# TUTORIAL
## Quick Start (macOS)
conda activate schmiede25
python app.py instrument_detector --live

## Quickstart (everywhere else)
source venv/bin/activate    (you may ofcourse use the vema of your choice as well)
python app.py instrument_detector --live

## OSC 
OSC output is automatically enabled in live mode and runs on localhost:9000.
It uses the following addresses and schema to send 3 packages, one for each topk prediction:
/instrument/top1, /instrument/top2, /instrument/top3 -> .../[ str(label), clamp_01(topk-score) ]

Be aware, OSC is blindly blasting on port 9000 and you'll need to decode ASCII into strings to read out the label/score pairs.

## Manual installation (Linux/Windows)
curl https://github.com/hideosnes/schmiede25-listening.git | scripts/setup.sh bash

### Install Miniconda or Anaconda first:
https://docs.conda.io/en/latest/miniconda.html

### Create conda environment
conda create -n myCondaName python=3.10
conda activate myCondaName

### Install scientific packages via conda (pre-compiled binaries)
conda install scipy librosa numpy
pip install torch torchaudio transformers huggingface-hub soundfile

If you are a masochist, go ahead and install pyaudio directly. Alternatively, it is strongly recommended to use Homebrew[https://brew.sh]
brew install portaudio
- after that -
pip install pyaudio

pip install -r requirements.txt

If you run into problems installing torch on a mac, kindly directly refer to the respective Docs / Github Issues of the library/project that causes the installation to fail.
If that doesn't help, you may as well attempt to contact apple customer-support about the issue. -_-

## Clone and run the project
git clone https://github.com/hideosnes/schmiede25-listening.git
cd schmiede25-listening
python app.py --list

# ARGUMENTS 
## Test available microphones
python app.py microphone_manager [OPTIONS]

Options:
  --list                    List all available microphones/input devices
  --test ID                 Test microphone by device ID (get ID from --list)
  --duration SECONDS        Test duration in seconds (default: 5)

### List all microphones
python app.py microphone_manager --list
#### Test microphone ID 0 for 10 seconds
python app.py microphone_manager --test 0 --duration 10

## Detect instruments
python app.py instrument_detector [OPTIONS]

Options:
  --model-path, -m PATH     Path to model directory (default: models/musical_instrument_detection)
  --from-hub                Force download model from HuggingFace Hub
  --hub-name NAME           Model ID on HuggingFace (default: dima806/musical_instrument_detection)
  --hf-token TOKEN          HuggingFace auth token for private models
  --test [FILE]             Quick test mode (default: test_wav/test.wav)
  --file, -f FILE           Audio file to analyze
  --sample-rate RATE        Target sample rate (default: 16000)
  --max-length-seconds SEC  Max audio length in seconds (default: 15)
  --topk K                  Show top-k predictions (default: 5)
  --live                    Run live mode with GUI and microphone input
  --chunk DURATION          Audio chunk duration for live mode in seconds (default: 1.0)

## Analyze a specific audio file
python app.py instrument_detector --file path/to/audio.wav

## Run live mode with GUI and OSC output
python app.py instrument_detector --live

## Live mode with 0.5 second chunks (more responsive)
python app.py instrument_detector --live --chunk 0.5

## Quick test with default test file
python app.py instrument_detector --test

## Show top 10 predictions instead of 3
python app.py instrument_detector --file audio.wav --topk 10

## Force download fresh model from HuggingFace
python app.py instrument_detector --from-hub --live

## MODEL-ZOO 
## Other audio Models
"MIT/ast-finetuned-audioset-10-10-0.4593" # Recognizes 527+ audio classes
"facebook/wav2vec2-large-xlsr-53-german"   # Different domain but more classes
"facebook/hubert-large-ls960-ft" # recognizes wooden and brass instruments
"microsoft/unispeech-sat-base-plus" # as with anything else by Microsoft: works only sometimes
"audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim" # solid, but very slow
## USAGE:
python app.py instrument_detector --hub-name "MIT/ast-finetuned-audioset-10-10-0.4593" --live

# ROADMAP & TODOs
## RAVE
### https://iil.is/news/ravemodels
RAVE (Realtime Audio Variational autoEncoder) learns an encoder which extracts compressed features from audio, and also a decoder which takes them back to sound. You can either use the decoder by itself as a unique kind of synthesizer, or run new audio through the encoder-decoder pair, transforming it to sound more like the training data.

## MIDI
### https://huggingface.co/santifiorino/SAO-Instrumental-Finetune
Create a MoE subset to generate organized midi-data-packages. These can be used to control e.g. hardware or software that expects MIDI inputs.

## Model Cascade
First model: Is this an instrument? (binary)
Second model: Which instrument family? (strings/brass/percussion/etc.)
Third model: Specific instrument within family

## Plugins
PureData
Isadora

## NM-library