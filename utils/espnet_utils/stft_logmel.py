from distutils.version import LooseVersion
from typing import Optional
from typing import Tuple
from typing import Union
import torch
from torch_complex.tensor import ComplexTensor
from .nets_utils import make_pad_mask
import numpy as np
import librosa

is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")
is_torch_1_7_plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")
is_torch_1_10_plus = LooseVersion(torch.__version__) >= LooseVersion("1.10")

def is_torch_complex_tensor(c):
    return is_torch_1_9_plus and torch.is_complex(c)

def is_complex(c):
    return isinstance(c, ComplexTensor) or is_torch_complex_tensor(c)

class Stft(torch.nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(torch, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window

    def extra_repr(self):
        return (
            f"n_fft={self.n_fft}, "
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
            f"normalized={self.normalized}, "
            f"onesided={self.onesided}"
        )

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """STFT forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2) or (Batch, Frames, Channels, Freq, 2)

        """
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        # NOTE(kamo):
        #   The default behaviour of torch.stft is compatible with librosa.stft
        #   about padding and scaling.
        #   Note that it's different from scipy.signal.stft

        # output: (Batch, Freq, Frames, 2=real_imag)
        # or (Batch, Channel, Freq, Frames, 2=real_imag)
        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(
                self.win_length, dtype=input.dtype, device=input.device
            )
        else:
            window = None

        # For the compatibility of ARM devices, which do not support
        # torch.stft() due to the lake of MKL.
        if input.is_cuda or torch.backends.mkl.is_available() or is_torch_1_10_plus:
            stft_kwargs = dict(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=self.center,
                window=window,
                normalized=self.normalized,
                onesided=self.onesided,
            )
            stft_kwargs["return_complex"] = True
            output = torch.stft(input.float(), **stft_kwargs)
            output = torch.view_as_real(output).type(input.dtype)
            #if is_torch_1_7_plus:
            #    stft_kwargs["return_complex"] = False
            #output = torch.stft(input, **stft_kwargs)
        else:
            if self.training:
                raise NotImplementedError(
                    "stft is implemented with librosa on this device, which does not "
                    "support the training mode."
                )

            # use stft_kwargs to flexibly control different PyTorch versions' kwargs
            stft_kwargs = dict(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=self.center,
                window=window,
                pad_mode="reflect",
            )

            if window is not None:
                # pad the given window to n_fft
                n_pad_left = (self.n_fft - window.shape[0]) // 2
                n_pad_right = self.n_fft - window.shape[0] - n_pad_left
                stft_kwargs["window"] = torch.cat(
                    [torch.zeros(n_pad_left), window, torch.zeros(n_pad_right)], 0
                ).numpy()
            else:
                win_length = (
                    self.win_length if self.win_length is not None else self.n_fft
                )
                stft_kwargs["window"] = torch.ones(win_length)

            output = []
            # iterate over istances in a batch
            for i, instance in enumerate(input):
                #raise NotImplementedError('Uncomment the import librosa statement above.')
                stft = librosa.stft(input[i].numpy(), **stft_kwargs)
                #stft = None
                output.append(torch.tensor(np.stack([stft.real, stft.imag], -1)))
            output = torch.stack(output, 0)
            if not self.onesided:
                len_conj = self.n_fft - output.shape[1]
                conj = output[:, 1 : 1 + len_conj].flip(1)
                conj[:, :, :, -1].data *= -1
                output = torch.cat([output, conj], 1)
            if self.normalized:
                output = output * (stft_kwargs["window"].shape[0] ** (-0.5))

        # output: (Batch, Freq, Frames, 2=real_imag)
        # -> (Batch, Frames, Freq, 2=real_imag)
        output = output.transpose(1, 2)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2), 2).transpose(
                1, 2
            )

        if ilens is not None:
            if self.center:
                pad = self.win_length // 2
                ilens = ilens + 2 * pad

            #olens = (ilens - self.win_length) // self.hop_length + 1
            olens = (
                torch.div(ilens - self.win_length, self.hop_length, rounding_mode="trunc")
                + 1
            )
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output, olens

    def inverse(
        self, input: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Inverse STFT.

        Args:
            input: Tensor(batch, T, F, 2) or ComplexTensor(batch, T, F)
            ilens: (batch,)
        Returns:
            wavs: (batch, samples)
            ilens: (batch,)
        """
        if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
            istft = torch.functional.istft
        else:
            try:
                import torchaudio
            except ImportError:
                raise ImportError(
                    "Please install torchaudio>=0.3.0 or use torch>=1.6.0"
                )

            if not hasattr(torchaudio.functional, "istft"):
                raise ImportError(
                    "Please install torchaudio>=0.3.0 or use torch>=1.6.0"
                )
            istft = torchaudio.functional.istft

        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            if is_complex(input):
                datatype = input.real.dtype
            else:
                datatype = input.dtype
            window = window_func(self.win_length, dtype=datatype, device=input.device)
        else:
            window = None

        if is_complex(input):
            input = torch.stack([input.real, input.imag], dim=-1)
        elif input.shape[-1] != 2:
            raise TypeError("Invalid input type")
        input = input.transpose(1, 2)

        wavs = istft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=ilens.max() if ilens is not None else ilens,
        )

        return wavs, ilens


class LogMel(torch.nn.Module):
    """Convert STFT to fbank feats

    The arguments is same as librosa.filters.mel

    Args:
        fs: number > 0 [scalar] sampling rate of the incoming signal
        n_fft: int > 0 [scalar] number of FFT components
        n_mels: int > 0 [scalar] number of Mel bands to generate
        fmin: float >= 0 [scalar] lowest frequency (in Hz)
        fmax: float >= 0 [scalar] highest frequency (in Hz).
            If `None`, use `fmax = fs / 2.0`
        htk: use HTK formula instead of Slaney
    """

    def __init__(
            self,
            fs: int = 16000,
            n_fft: int = 512,
            n_mels: int = 80,
            fmin: float = None,
            fmax: float = None,
            htk: bool = False,
            log_base: float = None,
    ):
        super().__init__()

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        _mel_options = dict(
            sr=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.mel_options = _mel_options
        self.log_base = log_base

        # Note(kamo): The mel matrix of librosa is different from kaldi.

        melmat = librosa.filters.mel(**_mel_options)
        # melmat: (D2, D1) -> (D1, D2)
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.mel_options.items())

    def forward(
            self,
            feat: torch.Tensor,
            ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        mel_feat = torch.matmul(feat, self.melmat)
        mel_feat = torch.clamp(mel_feat, min=1e-10)

        if self.log_base is None:
            logmel_feat = mel_feat.log()
        elif self.log_base == 2.0:
            logmel_feat = mel_feat.log2()
        elif self.log_base == 10.0:
            logmel_feat = mel_feat.log10()
        else:
            logmel_feat = mel_feat.log() / torch.log(self.log_base)

        # Zero padding
        if ilens is not None:
            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            ilens = feat.new_full(
                [feat.size(0)], fill_value=feat.size(1), dtype=torch.long
            )
        return logmel_feat, ilens
