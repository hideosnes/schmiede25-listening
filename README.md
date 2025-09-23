TUTORIAL:

## Quick Start

Get up and running with a single command:

```bash
curl -sSL https://github.com/hideosnes/schmiede25-listening/raw/main/scripts/setup.sh | bash


TODO:

# RAVE #
## https://iil.is/news/ravemodels ##
RAVE (Realtime Audio Variational autoEncoder) learns an encoder which extracts compressed features from audio, and also a decoder which takes them back to sound. You can either use the decoder by itself as a unique kind of synthesizer, or run new audio through the encoder-decoder pair, transforming it to sound more like the training data.

# MIDI #
## https://huggingface.co/santifiorino/SAO-Instrumental-Finetune ##
Create a MoE subset to generate organized midi-data-packages. These can be used to control e.g. hardware or software that expects MIDI inputs. 