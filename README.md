# TUTORIAL:

## Download


## Quick Start
### 1. Activate your environment
conda activate schmiede25

### 2. List available microphones
python app.py microphone_manager --list

### 3. Test your microphone
python app.py microphone_manager --test 0

### 4. Run live instrument detection
python app.py instrument_detector --live

# 5. For musicians: OSC output is automatically enabled in live mode
# Receives data on localhost:9000 with addresses:
# /instrument/top1, /instrument/top2, /instrument/top3
# /instrument/labels, /instrument/scores

## Manual installation
### Install Miniconda or Anaconda first:
https://docs.conda.io/en/latest/miniconda.html

### Create conda environment
conda create -n schmiede25 python=3.10
conda activate schmiede25

### Install scientific packages via conda (pre-compiled binaries)
conda install scipy librosa numpy

### Install remaining packages via pip
pip install torch torchaudio transformers huggingface-hub soundfile
brew install portaudio
pip install pyaudio
pip install -r requirements.txt

# Clone and run the project
git clone https://github.com/hideosnes/schmiede25-listening.git
cd schmiede25-listening
python app.py --list

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

## Show top 10 predictions instead of 5
python app.py instrument_detector --file audio.wav --topk 10

## Force download fresh model from HuggingFace
python app.py instrument_detector --from-hub --live

TODO:

# RAVE #
## https://iil.is/news/ravemodels ##
RAVE (Realtime Audio Variational autoEncoder) learns an encoder which extracts compressed features from audio, and also a decoder which takes them back to sound. You can either use the decoder by itself as a unique kind of synthesizer, or run new audio through the encoder-decoder pair, transforming it to sound more like the training data.

# MIDI #
## https://huggingface.co/santifiorino/SAO-Instrumental-Finetune ##
Create a MoE subset to generate organized midi-data-packages. These can be used to control e.g. hardware or software that expects MIDI inputs. 