# About
Compact and pure python/pytorch re-implementation of the NeLF ASR model.
Private repo: do not distribute the code and or models.

# Usage
## 1. Recreate Python environment 
Python version: 3.12
pip install -r requirements.txt

## 2. Download models

URL: 
https://huggingface.co/nelfproject/NeLF_S2T_Pytorch
or
https://huggingface.co/Jakobkee/NeLF_S2T_Pytorch

## 3. Set parameters:

Set your parameters and local setup options in test_decode_with_vad.py.

Main options:
 -  local paths: downloaded model directory, output folder, audio data
 -  desired outputs: only encoder outputs, verbatim decoder outputs, subtitle decoder outputs
 -  device settings


## 4. Run!
python test_decode_with_vad.py

# Contact
jakob(dot)poncelet(at)esat(dot)kuleuven(dot)be
