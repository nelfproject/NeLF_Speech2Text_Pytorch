import os
import argparse
from typing import Union, Tuple
from pathlib import Path
import yaml
import torch
from torch import nn
from itertools import groupby
import soundfile

from .espnet_utils.text import TokenIDConverter, build_tokenizer
from .espnet_utils.conformer_encoder import ConformerEncoder
from .espnet_utils.transformer_encoder import TransformerEncoder
from .espnet_utils.multi_transformer_decoder import MultiTransformerDecoder
from .espnet_utils.specaugment import SpecAug
from .espnet_utils.frontend import DefaultFrontend
from .espnet_utils.utterance_mvn import UtteranceMVN
from .espnet_utils.global_mvn import GlobalMVN
from .espnet_utils.ctc import CTC
from .espnet_utils.scorers import CTCPrefixScorer, LengthBonus
from .espnet_utils.beam_search import BeamSearch
from .espnet_utils.batch_beam_search import BatchBeamSearch
from .espnet_utils.nets_utils import set_all_random_seed

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
    subtitle_ctc: bool = False,
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
        subtitle_ctc: Generate CTC predictions from the subtitle encoder outputs.
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
    model = build_model(args, ctc=ctc, subtitle_ctc=subtitle_ctc, verbatim=verbatim, subtitle=subtitle, vocab_size=vocab_size, tokenizer=tokenizer, converter=converter, token_list=token_list, stats_file=stats_file)
    model.to(device)

    # Load model checkpoint
    if device == "cuda":
        device = f"cuda:{torch.cuda.current_device()}"
    result = model.load_state_dict(torch.load(model_file, map_location=device), strict=False)

    if not training:
        model.eval()

    #if verbatim or subtitle:
    #    model.prepare_for_beam_search(token_list)

    return model, args

def build_model(args: argparse.Namespace, ctc=False, subtitle_ctc=False, verbatim=False, subtitle=False, vocab_size=None, tokenizer=None, converter=None, token_list=None, stats_file=None):
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

    # Subtitle Encoder
    if verbatim or subtitle:
        subtitle_encoder = TransformerEncoder(input_size=encoder.output_size(), **args.subtitle_encoder_conf)
    else:
        subtitle_encoder = None

    # Verbatim CTC
    if ctc or verbatim:
        ctc = CTC(
            odim=vocab_size,
            encoder_output_sizse=encoder.output_size(),
            **args.ctc_conf,
        )
    else:
        ctc = None

    # Subtitle CTC
    if subtitle_ctc or subtitle:
        subtitle_ctc = CTC(
            odim=vocab_size,
            encoder_output_sizse=subtitle_encoder.output_size(),
            **args.ctc_conf,
        )

    # Verbatim Decoder
    if verbatim:
        verbatim_decoder = MultiTransformerDecoder(vocab_size=vocab_size, encoder_output_size=encoder.output_size(), **args.decoder_conf)
    else:
        verbatim_decoder = None

    # Subtitle Decoder
    if subtitle:
        subtitle_decoder = MultiTransformerDecoder(vocab_size=vocab_size, encoder_output_size=encoder.output_size(), **args.subtitle_decoder_conf)
    else:
        subtitle_decoder = None

    # overwrite: this is not trained
    subtitle_ctc = None

    # Build model
    model = NeLFModel(
        specaug=specaug,
        frontend=frontend,
        normalize=normalize,
        encoder=encoder,
        subtitle_encoder=subtitle_encoder,
        verbatim_decoder=verbatim_decoder,
        subtitle_decoder=subtitle_decoder,
        ctc=ctc,
        subtitle_ctc=subtitle_ctc,
        tokenizer=tokenizer,
        converter=converter,
        token_list=token_list,
    )
    return model


class NeLFModel(nn.Module):
    """ NeLF Model """
    def __init__(
        self,
        specaug,
        frontend,
        normalize,
        encoder,
        subtitle_encoder,
        verbatim_decoder,
        subtitle_decoder,
        ctc,
        subtitle_ctc,
        tokenizer,
        converter,
        token_list,
        blank=0,
    ):
        super().__init__()
        self.specaug = specaug
        self.frontend = frontend
        self.normalize = normalize
        self.encoder = encoder
        self.subtitle_encoder = subtitle_encoder
        self.decoder = verbatim_decoder
        self.subtitle_decoder = subtitle_decoder
        self.ctc = ctc
        self.subtitle_ctc = subtitle_ctc
        self.tokenizer = tokenizer
        self.converter = converter
        self.token_list = token_list
        self.blank = blank

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Union[None, torch.Tensor], Union[None, torch.Tensor]]:
        """Frontend + Encoders Only for Feature Extraction

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

    def decode(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            ctc: bool = True,
            verbatim: bool = False,
            subtitle: bool = False,
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

            # Subtitle Encoder
            if self.subtitle_encoder is not None:
                sub_encoder_out, sub_encoder_out_lens, _ = self.subtitle_encoder(
                        encoder_out, encoder_out_lens)

                if isinstance(sub_encoder_out, tuple):
                    sub_encoder_out = sub_encoder_out[0]
            
            else:
                sub_encoder_out, sub_encoder_out_lens = None, None

        results = {}

        if ctc:
            ctc_results = self.decode_ctc(encoder_out, mode="verbatim")
            results["ctc"] = ctc_results

        if ctc and self.subtitle_ctc is not None:
            subtitle_ctc_results = self.decode_ctc(sub_encoder_out, mode="subtitle")
            results["subtitle_ctc"] = subtitle_ctc_results

        if verbatim:
            verbatim_results = self.decode_full(encoder_out, sub_encoder_out, mode="verbatim")
            results["verbatim"] = verbatim_results

        if subtitle:
            subtitle_results = self.decode_full(encoder_out, sub_encoder_out, mode="subtitle")
            results["subtitle"] = subtitle_results

        return results

    def decode_ctc(self, encoder_feats, mode="verbatim"):
        """ 
        Generate ctc token predictions from encoder output 
        and convert to sentence 
        """
        if mode == "verbatim":
            ctc_preds = self.ctc.argmax(encoder_feats)
        elif mode == "subtitle":
            ctc_preds = self.subtitle_ctc.argmax(encoder_feats)

        res = []
        for seq in ctc_preds:
            seq_list = seq.tolist()

            # collapse and remove blanks
            collapsed = [k for k, _ in groupby(seq_list) if k != self.blank]

            # Change integer-ids to tokens
            tokens = self.converter.ids2tokens(collapsed)

            # Convert to text
            text = self.tokenizer.tokens2text(tokens)

            res.append(text)
        return res

    def decode_full(self, encoder_out, sub_encoder_out, mode="verbatim"):
        if mode == "verbatim":
            decoder_inputs = (encoder_out.squeeze(0), sub_encoder_out.squeeze(0))
            search_func = self.verbatim_beam_search
        elif mode == "subtitle":
            decoder_inputs = (sub_encoder_out.squeeze(0), encoder_out.squeeze(0))
            search_func = self.subtitle_beam_search
        else:
            raise ValueError(f"mode should be either verbatim or subtitle, got {mode}")
        
        nbest_hyps = search_func(
            x = decoder_inputs,
            maxlenratio = self.beam_search_conf["maxlenratio"],
            minlenratio = self.beam_search_conf["minlenratio"],
        )[:self.beam_search_conf["nbest"]]

        results = []
        for hyp in nbest_hyps:
            # remove sos/eos and get results
            token_int = hyp.yseq[1:-1].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != self.blank, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            
            text = text.strip()

            if text and mode == 'subtitle':
                if text[0].isalpha():
                    text = text[0].upper() + text[1:]
            results.append(text)
        return results    
    
    def prepare_for_beam_search(self, nbest=1, beam_size=20, ctc_weight=0.3, subtitle_ctc_weight=0.0, length_penalty=0.0, subtitle_length_penalty=0.0, minlenratio=0.0, maxlenratio=0.0, normalize_length_verbatim=False, normalize_length_subtitle=False):
        """ Prepare all modules for beam search """
        set_all_random_seed(42)

        token_list = self.converter.token_list
        eos_token = token_list.index('<sos/eos>')
        sos_token = token_list.index('<sos/eos>')

        self.beam_search_conf = {
            "nbest": nbest,
            "beam_size": beam_size,
            "ctc_weight": ctc_weight,
            "subtitle_ctc_weight": subtitle_ctc_weight,
            "length_penalty": length_penalty,
            "subtitle_length_penalty": subtitle_length_penalty,
            "maxlenratio": maxlenratio,
            "minlenratio": minlenratio,
            "eos_token": eos_token,
            "sos_token": sos_token,
        }

        if self.decoder is not None:
            verbatim_scorers = {
                "decoder": self.decoder,
                "ctc": CTCPrefixScorer(ctc=self.ctc, eos=eos_token),
                "length_bonus": LengthBonus(len(token_list)),
            }
        
            verbatim_weights = {
                "decoder": 1 - ctc_weight,
                "ctc_weight": ctc_weight,
                "length_bonus": length_penalty,
            }

            verbatim_beam_search = BeamSearch(
                beam_size=beam_size,
                weights=verbatim_weights,
                scorers=verbatim_scorers,
                sos=sos_token,
                eos=eos_token,
                vocab_size=len(token_list),
                token_list=token_list,
                pre_beam_score_key=None if ctc_weight == 1.0 else "full",
                normalize_length=normalize_length_verbatim,
            )
            verbatim_beam_search.__class__ = BatchBeamSearch

            self.verbatim_beam_search = verbatim_beam_search

        if self.subtitle_decoder is not None:
            subtitle_scorers = {
                "decoder": self.subtitle_decoder,
                "length_bonus": LengthBonus(len(token_list)),
            }
        
            subtitle_weights = {
                "decoder": 1 - subtitle_ctc_weight,
                "ctc_weight": subtitle_ctc_weight,
                "length_bonus": subtitle_length_penalty,
            }

            subtitle_beam_search = BeamSearch(
                beam_size=beam_size,
                weights=subtitle_weights,
                scorers=subtitle_scorers,
                sos=sos_token,
                eos=eos_token,
                vocab_size=len(token_list),
                token_list=token_list,
                pre_beam_score_key=None if subtitle_ctc_weight == 1.0 else "full",
                normalize_length=normalize_length_subtitle,
            )
            subtitle_beam_search.__class__ = BatchBeamSearch
            
            self.subtitle_beam_search = subtitle_beam_search

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

def load_nelf_model(model_dir=None, ctc=True, subtitle_ctc=False, verbatim=False, subtitle=False, device="cuda"):
    subtitle_ctc = False  # not trained, overwrite

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

    model, args = build_model_from_file(asr_config, asr_model, tokens, bpe_model, ctc, subtitle_ctc, verbatim, subtitle, stats_file, device)
    return model, args

