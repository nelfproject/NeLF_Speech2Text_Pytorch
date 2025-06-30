import os, sys
from utils.espnet_model import load_wav, load_nelf_model
from utils.vad_model import load_vad, load_audio
from utils.postprocessing import format_output, write_output_to_file, merge_batch_outputs
import torch
import time
import logging
import warnings

#### SETTINGS ####
# Device to use: cuda / cpu
device = "cpu"

# Number of CPU cores to use
num_cpus = 8

# Directory where you installed the model
model_dir = "/esat/audioslave/jponcele/models/ASR_subtitles_pytorch"

# Select desired model outputs
encoder_outputs = True
verbatim_decoder_outputs = False
subtitle_decoder_outputs = False

# Options for decoder beam search
decode_conf = {
    "nbest": 1,  # return the top N final predictions
    "beam_size": 20,  # amount of alternatives to keep during beam search
    "minlenratio": 0.2,  # minimum ratio of speech frames vs predicted text frames
}

# Decode all segments at the same time (only works for encoder-only)
batch_decode = False  #True

# Settings related to the VAD model
VAD_model_dir = "/esat/audioslave/jponcele/models/ASR_subtitles_pytorch/VAD"
VAD_use_onnx = False  # if True, use ONNX model, else use torch.jit model

max_segment_length = 15.0  # maximum length of one speech segment
min_segment_length = 3.0  # minimum length of a speech segment
max_pause = 2.0  # if pause between segments is smaller than this, try to merge segments

min_speech_duration_ms = 250  # minimum amount of speech in ms to trigger VAD
min_silence_duration_ms = 250  # minimum amount of silence in ms to trigger VAD
speech_pad_ms = 60  # pad speech segments on each side to definitely not cut off speech
vad_threshold = 0.5  # speech vs silence classification

# Directory with audio examples
wav_dir = "./audio_examples/long_wavs"

# Output related
output_add_timing = True  # add timing of segments to output
verbose_output = False  # print output to terminal
write_output = False # print output to file
output_dir = "./output/long_wavs"
###################

# Sampling rate (do not change!)
sr = 16000

# Set the number of cpus
torch.set_num_threads(num_cpus)

# Suppress unnecessary logging
logging.getLogger().setLevel(logging.CRITICAL)  #DEBUG)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the model
print('# Loading Speech-to-Text Model')
model, args = load_nelf_model(
    model_dir=model_dir,
    ctc=encoder_outputs, 
    subtitle_ctc=encoder_outputs,
    verbatim=verbatim_decoder_outputs,
    subtitle=subtitle_decoder_outputs,
    device=device)

# Initialize the beam search for decoding with decoders
if verbatim_decoder_outputs or subtitle_decoder_outputs:
    print('# Preparing decoders')
    model.prepare_for_beam_search(**decode_conf)

    if batch_decode:
        print('NOTE: You have set batch_decode=True but this only works for encoder-only models. Defaulting back to batch_decode=False')
        batch_decode = False

# Set inference mode
model.eval()

# Load the VAD model
print('# Loading VAD model')
vad_model = load_vad(VAD_model_dir, onnx=VAD_use_onnx, device=device)

# Disable gradients for inference
torch.set_grad_enabled(False)

# Loop over wav files
print(f"# Processing all wav files in {wav_dir}")
for wav_file in sorted(os.listdir(wav_dir)):
    if not wav_file.endswith('.wav'):
        continue

    print(f"--- Audio file: {wav_file} ---")
    start_time = time.time()

    # load wav file
    wav = load_audio(os.path.join(wav_dir, wav_file), sampling_rate=sr, device=device)
    #wav, lens = load_wav(os.path.join(wav_dir, wav_file), device=device)
    
    # apply VAD
    speech_timestamps = vad_model.get_speech_timestamps(
        wav,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        threshold=vad_threshold,
    )

    # merge VAD segments
    speech_segments = vad_model.merge_segments(
        speech_timestamps,
        max_pause,
        max_segment_length,
        min_segment_length,
        mode="hierarchical",
    )

    # decode speech segments to text
    if batch_decode:
        # combine all segments and decode simultaneously
        wav_batch, wav_lens = [], []
        for seg in speech_segments:
            wav_seg = wav[int(seg['start']*sr):int(seg['end']*sr)].to(device=device)
            wav_len = torch.tensor([wav_seg.shape[0],], dtype=torch.long, device=device)
            wav_batch.append(wav_seg)
            wav_lens.append(wav_len)
    
        wav_batch = torch.nn.utils.rnn.pad_sequence(wav_batch, batch_first=True)
        wav_lens = torch.cat(wav_lens)
        res = model.decode(wav_batch, wav_lens, ctc=encoder_outputs, verbatim=verbatim_decoder_outputs, subtitle=subtitle_decoder_outputs)
    
        out = format_output(res, speech_segments, add_timing=output_add_timing, batch=True, verbose=verbose_output)
    else:
        # decode segment by segment
        batch_out = []
        for seg in speech_segments:
            wav_seg = wav[int(seg['start']*sr):int(seg['end']*sr)].unsqueeze(0).to(device=device)
            wav_len = torch.tensor([wav_seg.shape[1],], dtype=torch.long, device=device)
            res = model.decode(wav_seg, wav_len, ctc=encoder_outputs, verbatim=verbatim_decoder_outputs, subtitle=subtitle_decoder_outputs)
            
            out = format_output(res, [seg], add_timing=output_add_timing, verbose=verbose_output)
            batch_out.append(out)
        out = merge_batch_outputs(batch_out)

    if write_output:
        os.makedirs(output_dir, exist_ok=True)
        write_output_to_file(out, output_dir, wav_file[:-4])

    end_time = time.time()
    print('    -- Processing %.2f seconds of audio took %.2f seconds' % (len(wav)/sr, (end_time - start_time)))

