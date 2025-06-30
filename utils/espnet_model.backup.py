import os
import argparse
from typing import Union, Tuple
from pathlib import Path
import yaml
import torch
from torch import nn
from itertools import groupby
import kaldiio
import soundfile

from .espnet_utils.text import TokenIDConverter, build_tokenizer
from .espnet_utils.conformer_encoder import ConformerEncoder
from .espnet_utils.transformer_encoder import TransformerEncoder
from .espnet_utils.specaugment import SpecAug
from .espnet_utils.frontend import DefaultFrontend
from .espnet_utils.utterance_mvn import UtteranceMVN
from .espnet_utils.global_mvn import GlobalMVN
from .espnet_utils.ctc import CTC

from contextlib import contextmanager
from distutils.version import LooseVersion

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

def build_model_from_file(
    config_file: Union[Path, str],
    model_file: Union[Path, str],
    token_list: Union[Path, str],
    bpemodel: Union[Path, str],
    ctc: bool = False,
    verbatim: bool = False,
    subtitle: bool = False,
    stats_file: Union[Path, str] = None,
    device: str = "cpu",
    training: bool = False,
):
    """
    Build model from files.

    Input args:
        config_file: The yaml file saved when training.
        model_file: The model file saved when training.
        token_list: The token list used for training.
        bpemodel: The BPE model used for training.
        ctc: Generate CTC predictions from the encoder outputs.
        verbatim: Generate verbatim predictions from verbatim decoder outputs.
        subtitle: Generate subtitle predictions from subtitle decoder outputs.
        device: Device type, "cpu", "cuda", or "cuda:N".
        training: If model is used for training or for inference.
    Return args:
        model: The loaded model
        args: The loaded arguments
    """
    
    # Load config file
    config_file = Path(config_file)
    with config_file.open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args = argparse.Namespace(**args)

    # Load token list
    token_list = Path(token_list)
    with token_list.open("r", encoding="utf-8") as f:
        token_list = [line.rstrip() for line in f]
    vocab_size = len(token_list)

    # Load bpe model
    tokenizer = build_tokenizer(token_type="bpe", bpemodel=bpemodel)
    converter = TokenIDConverter(token_list=token_list)

    # Build ASR model
    model = build_model(args, ctc=ctc, verbatim=verbatim, subtitle=subtitle, vocab_size=vocab_size, tokenizer=tokenizer, converter=converter, stats_file=stats_file)
    model.to(device)

    # Load model checkpoint
    if device == "cuda":
        device = f"cuda:{torch.cuda.current_device()}"
    result = model.load_state_dict(torch.load(model_file, map_location=device), strict=False)

    if not training:
        model.eval()

    return model, args

def build_model(args: argparse.Namespace, ctc=False, verbatim=False, subtitle=False, vocab_size=None, tokenizer=None, converter=None, stats_file=None):
    """ Build NeLF Model """
    # Data augmentation for spectrogram
    specaug = SpecAug(**args.specaug_conf)

    if args.frontend is not None:
        assert args.frontend == "default", "Only installed DefaultFrontend right now"
        frontend = DefaultFrontend(**args.frontend_conf)
        input_size = frontend.output_size()
    else:
        frontend = None
        input_size = args.input_size

    # Normalization layer
    if args.normalize == "global_mvn":
        if stats_file is not None:
            args.normalize_conf["stats_file"] = stats_file
        normalize = GlobalMVN(**args.normalize_conf)
    else:
        normalize = UtteranceMVN(**args.normalize_conf)

    # Verbatim Encoder
    encoder = ConformerEncoder(input_size=input_size, **args.encoder_conf)

    if ctc and vocab_size is not None:
        ctc = CTC(
            odim=vocab_size,
            encoder_output_sizse=encoder.output_size(),
            **args.ctc_conf,
        )
    else:
        ctc = None

    # Subtitle Encoder
    if verbatim or subtitle:
        subtitle_encoder = TransformerEncoder(input_size=encoder.output_size(), **args.subtitle_encoder_conf)
    else:
        subtitle_encoder = None

    # Build model
    model = NeLFEncoder(
        specaug=specaug,
        frontend=frontend,
        normalize=normalize,
        encoder=encoder,
        subtitle_encoder=subtitle_encoder,
        ctc=ctc,
        tokenizer=tokenizer,
        converter=converter,
    )
    return model


class NeLFEncoder(nn.Module):
    """ NeLF Model speech encoder """
    def __init__(
        self,
        specaug,
        frontend,
        normalize,
        encoder,
        subtitle_encoder,
        ctc,
        tokenizer,
        converter,
    ):
        super().__init__()
        self.specaug = specaug
        self.frontend = frontend
        self.normalize = normalize
        self.encoder = encoder
        self.subtitle_encoder = subtitle_encoder
        self.ctc = ctc
        self.tokenizer = tokenizer
        self.converter = converter

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Union[None, torch.Tensor], Union[None, torch.Tensor]]:
        """Frontend + Encoders

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
        """
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape)

        with autocast(False):
            if self.frontend is not None:
                feats, feats_lengths = self.frontend(speech, speech_lengths)
            else:
                feats, feats_lengths = speech, speech_lengths

            # Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

            # Verbatim Encoder
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]

            # Subtitle Encoder
            if self.subtitle_encoder is not None:
                sub_encoder_out, sub_encoder_out_lens, _ = self.subtitle_encoder(
                        encoder_out, encoder_out_lens)
            else:
                sub_encoder_out, sub_encoder_out_lens = None, None

        return encoder_out, encoder_out_lens, sub_encoder_out, sub_encoder_out_lens

    def decode_ctc(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            blank: int = 0,
    ) -> Tuple[torch.Tensor]:
        """Frontend + Encoders

                Args:
                    speech: (Batch, Length, ...)
                    speech_lengths: (Batch,)
                """
        assert (
                speech.shape[0]
                == speech_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape)

        assert self.ctc is not None, "self.ctc is None"

        with autocast(False):
            if self.frontend is not None:
                feats, feats_lengths = self.frontend(speech, speech_lengths)
            else:
                feats, feats_lengths = speech, speech_lengths

            # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

            # Verbatim Encoder
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]

            # ctc predictions
            ctc_preds = self.ctc.argmax(encoder_out)

        # convert to text
        results = []
        for seq in ctc_preds:
            seq_list = seq.tolist()

            # collapse and remove blanks
            collapsed = [k for k, _ in groupby(seq_list) if k != blank]

            # Change integer-ids to tokens
            tokens = self.converter.ids2tokens(collapsed)

            # Convert to text
            text = self.tokenizer.tokens2text(tokens)

            results.append(text)

        return results

def load_wav(wav_file, device="cuda"):
    """
    Load a sample wav file and return the wav tensor and the length tensor
    """
    if not os.path.exists(wav_file):
        raise FileNotFoundError(f"The file {wav_file} does not exist.")
    if not wav_file.endswith('.wav'):
        raise ValueError(f"Expected a .wav file, but got {wav_file}.")
        
    wav, sr = soundfile.read(wav_file)
    
    assert sr == 16000, f"Expected a sampling rate of 16000, but got {sr}"

    wav = torch.tensor(wav, device=device, dtype=torch.float32).unsqueeze(0)
    wav_len = torch.tensor([wav.shape[1],], dtype=torch.long, device=device)
    return wav, wav_len

def load_nelf_model(model_dir=None, ctc=True, verbatim=False, subtitle=False, device="cuda"):
    if model_dir is None:
        model_dir = "/esat/audioslave/jponcele/models/ASR_subtitles_pytorch"

    asr_model = f"{model_dir}/model.pth"
    asr_config = f"{model_dir}/train.yaml"
    tokens = f"{model_dir}/tokens.txt"
    bpe_model = f"{model_dir}/bpe.model"
    stats_file = f"{model_dir}/stats.npz"

    if not os.path.exists(stats_file):
        print('No feature stats file is found.')
        stats_file = None

    assert os.path.exists(asr_model), f"Could not find saved model checkpoint 'model.pth' in {model_dir}"
    assert os.path.exists(asr_config), f"Could not find saved model config 'train.yaml' in {model_dir}"
    assert os.path.exists(tokens), f"Could not find saved token list 'tokens.txt' in {model_dir}"
    assert os.path.exists(bpe_model), f"Could not find saved bpe model 'bpe.model' in {model_dir}"

    model, args = build_model_from_file(asr_config, asr_model, tokens, bpe_model, ctc, verbatim, subtitle, stats_file, device)
    return model, args

