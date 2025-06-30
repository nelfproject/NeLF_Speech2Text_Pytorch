import copy
from typing import Optional
from typing import Tuple
from typing import Union
from typing import List

import humanfriendly
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_complex.tensor import ComplexTensor

from .stft_logmel import Stft, LogMel
from .nets_utils import get_default_kwargs

class Frontend(nn.Module):
    def __init__(
        self,
        idim: int,
        # WPE options
        use_wpe: bool = False,
        wtype: str = "blstmp",
        wlayers: int = 3,
        wunits: int = 300,
        wprojs: int = 320,
        wdropout_rate: float = 0.0,
        taps: int = 5,
        delay: int = 3,
        use_dnn_mask_for_wpe: bool = True,
        # Beamformer options
        use_beamformer: bool = False,
        btype: str = "blstmp",
        blayers: int = 3,
        bunits: int = 300,
        bprojs: int = 320,
        bnmask: int = 2,
        badim: int = 320,
        ref_channel: int = -1,
        bdropout_rate=0.0,
    ):
        super().__init__()

        self.use_beamformer = use_beamformer
        self.use_wpe = use_wpe
        self.use_dnn_mask_for_wpe = use_dnn_mask_for_wpe
        # use frontend for all the data,
        # e.g. in the case of multi-speaker speech separation
        self.use_frontend_for_all = bnmask > 2

    def forward(
        self, x: ComplexTensor, ilens: Union[torch.LongTensor, np.ndarray, List[int]]
    ) -> Tuple[ComplexTensor, torch.LongTensor, Optional[ComplexTensor]]:
        assert len(x) == len(ilens), (len(x), len(ilens))
        # (B, T, F) or (B, T, C, F)
        if x.dim() not in (3, 4):
            raise ValueError(f"Input dim must be 3 or 4: {x.dim()}")
        if not torch.is_tensor(ilens):
            ilens = torch.from_numpy(np.asarray(ilens)).to(x.device)

        mask = None
        h = x
        if h.dim() == 4:
            if self.training:
                choices = [(False, False)] if not self.use_frontend_for_all else []
                if self.use_wpe:
                    choices.append((True, False))

                if self.use_beamformer:
                    choices.append((False, True))

                use_wpe, use_beamformer = choices[np.random.randint(len(choices))]

            else:
                use_wpe = self.use_wpe
                use_beamformer = self.use_beamformer

            # 1. WPE
            if use_wpe:
                # h: (B, T, C, F) -> h: (B, T, C, F)
                h, ilens, mask = self.wpe(h, ilens)

            # 2. Beamformer
            if use_beamformer:
                # h: (B, T, C, F) -> h: (B, T, F)
                h, ilens, mask = self.beamformer(h, ilens)

        return h, ilens, mask


def frontend_for(args, idim):
    return Frontend(
        idim=idim,
        # WPE options
        use_wpe=args.use_wpe,
        wtype=args.wtype,
        wlayers=args.wlayers,
        wunits=args.wunits,
        wprojs=args.wprojs,
        wdropout_rate=args.wdropout_rate,
        taps=args.wpe_taps,
        delay=args.wpe_delay,
        use_dnn_mask_for_wpe=args.use_dnn_mask_for_wpe,
        # Beamformer options
        use_beamformer=args.use_beamformer,
        btype=args.btype,
        blayers=args.blayers,
        bunits=args.bunits,
        bprojs=args.bprojs,
        bnmask=args.bnmask,
        badim=args.badim,
        ref_channel=args.ref_channel,
        bdropout_rate=args.bdropout_rate,
    )


class DefaultFrontend(torch.nn.Module):
    """Conventional frontend structure for ASR.

    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Mel-Fbank -> CMVN
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
        add_dummy_pitch: bool = False,
    ):
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)

        if apply_stft:
            self.stft = Stft(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=center,
                window=window,
                normalized=normalized,
                onesided=onesided,
            )
        else:
            self.stft = None
        self.apply_stft = apply_stft

        if frontend_conf is not None:
            self.frontend = Frontend(idim=n_fft // 2 + 1, **frontend_conf)
        else:
            self.frontend = None

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.n_mels = n_mels
        self.frontend_type = "default"
        self.add_dummy_pitch = add_dummy_pitch

    def output_size(self) -> int:
        if self.add_dummy_pitch:
            return int(self.n_mels + 3)
        return self.n_mels

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        if self.stft is not None:
            input_stft, feats_lens = self._compute_stft(input, input_lengths)
        else:
            input_stft = ComplexTensor(input[..., 0], input[..., 1])
            feats_lens = input_lengths
        # 2. [Option] Speech enhancement
        if self.frontend is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, Length, [Channel], Freq)
            input_stft, _, mask = self.frontend(input_stft, feats_lens)

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input_stft.size(2))
                input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real ** 2 + input_stft.imag ** 2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)

        if self.add_dummy_pitch:
            input_feats = F.pad(input_feats, (0,3), mode='constant', value=0.0)

        return input_feats, feats_lens

    def _compute_stft(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens



