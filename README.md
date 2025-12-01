# About
This is the codebase with compact and pure python/pytorch re-implementation of the NeLF ASR model for Flemish speech recognition with extensive finetuning.
This codebase is required to use the pre-trained ASR model from https://huggingface.co/nelfproject/NeLF_S2T_Pytorch.

For more information about Flemish Speech recognition in the NeLF project or contact details, visit our website: https://nelfproject.be 

# Usage
## 1. Recreate Python environment 
Python version: 3.12
pip install -r requirements.txt

## 2. Download models

URL: 
https://huggingface.co/nelfproject/NeLF_S2T_Pytorch

## 3. Set parameters:

Set your parameters and local setup options in test_decode_with_vad.py.

Main options:
 -  local paths: downloaded model directory, output folder, audio data
 -  desired outputs: only encoder outputs, verbatim decoder outputs, subtitle decoder outputs
 -  device settings


## 4. Run!
python test_decode_with_vad.py

# License
This codebase and the related models are provided with a Creative Commons Non-Commercial license.

# Research paper
If you use our code and models, please consider citing our research paper.

```bibtex
@article{poncelet2024,
    author = "Poncelet, Jakob and Van hamme, Hugo",
    title = "Leveraging Broadcast Media Subtitle Transcripts for Automatic Speech Recognition and Subtitling",
    year={2024},
    journal={arXiv preprint arXiv:2502.03212},
    url = {https://arxiv.org/abs/2502.03212}
}
```
