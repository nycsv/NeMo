# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from nemo.core import NeuralModule
from nemo.core.classes import Exportable, NeuralModule, typecheck
from nemo.core.neural_types import LabelsType, NeuralType, SpectrogramType


class RandomProjectionVectorQuantizer(NeuralModule, Exportable):
    DIST_FN_LIST = ["l2", "cosine"]

    def __init__(
        self,
        feat_in: int,
        code_dim: int,
        num_classes: int,
        num_books: int,
        dist_fn: str = "cosine",
        time_ahead: bool = False,
        freeze: bool = True,
        squeeze_single: bool = False,
        combine_time_steps: int = 1,
    ):
        """Vector quantization using random projection proposed in BEST-RQ paper:
        'Self-Supervised Learning with Random-Projection Quantizer for Speech Recognition'

         Args:
            feat_in: input feature dimension
            code_dim: dimension of the codebook features
            num_classes: number of classes
            num_books: number of codebooks
            dist_fn: distance function to use, one of "l2" or "cosine"
            time_ahead: if Ture, the input is of shape (B, T, D), otherwise (B, D, T)
            freeze: whether to freeze the projection matrix
            squeeze_single: if True, squeeze codebook dimension if num_books is 1
        """
        super().__init__()

        if dist_fn not in self.DIST_FN_LIST:
            raise ValueError(f"Unknown distance function {dist_fn}, must be one of {self.DIST_FN_LIST}")

        self.feat_in = feat_in
        self.code_dim = code_dim
        self.num_classes = num_classes
        self.num_books = num_books
        self.dist_fn = dist_fn
        self.time_ahead = time_ahead
        self.squeeze_single = squeeze_single
        self.combine_time_steps = combine_time_steps

        # (B, T, D) -> (B, T, num_books, code_dim)
        self.proj = nn.Linear(self.feat_in * combine_time_steps, self.num_books * self.code_dim, bias=False)
        torch.nn.init.xavier_normal_(self.proj.weight)

        # (num_books, num_classes, hid_dim)
        codebooks = torch.randn(self.num_books, self.num_classes, self.code_dim).double()
        torch.nn.init.normal_(codebooks, mean=0, std=1)
        codebooks = F.normalize(codebooks, dim=-1)
        self.codebooks = nn.Parameter(codebooks)
        if freeze:
            self.freeze()

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        if self.time_ahead:
            return {"input_signal": NeuralType(('B', 'T', 'D'), SpectrogramType())}
        return {"input_signal": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        if self.time_ahead:
            if self.num_books == 1 and self.squeeze_single:
                return {
                    "xq": NeuralType(('B', 'T', 'D'), SpectrogramType()),
                    "xid": NeuralType(('B', 'T'), LabelsType()),
                }
            return {
                "xq": NeuralType(('B', 'T', 'D', 'H'), SpectrogramType()),
                "xid": NeuralType(('B', 'T', 'H'), LabelsType()),
            }
        if self.num_books == 1 and self.squeeze_single:
            return {
                "xq": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "xid": NeuralType(('B', 'T'), LabelsType()),
            }
        return {
            "xq": NeuralType(('B', 'D', 'T', 'H'), SpectrogramType()),
            "xid": NeuralType(('B', 'T', 'H'), LabelsType()),
        }

    @typecheck()
    def forward(self, input_signal):
        """
        Args:
            input_signal: input features of shape (B, T, D) or (B, D, T)
        Returns:
            xq: quantized features of shape (B, T, D, N) or (B, D, T, N)
            xid: quantized tokens of shape (B, T, N)
        """
        if not self.time_ahead:
            # (B, D, T) -> (B, T, D)
            input_signal = input_signal.transpose(1, 2)

        B, T, _ = input_signal.size()

        if self.combine_time_steps > 1:
            input_signal = input_signal.contiguous().reshape(B, T // self.combine_time_steps, -1)
            T = T // self.combine_time_steps

        # (B, T, D) -> (B, T, num_books*code_dim)
        x = self.proj(input_signal)

        # normalize each feature vector
        # (B, T, num_books*code_dim) -> (B, T, num_books, code_dim)
        x = F.normalize(x.view(B, T, self.num_books, self.code_dim), dim=-1)

        # get tokens (xid) of shape (B, T, num_books)
        if self.dist_fn == "cosine":
            # (B, T, num_books, code_dim) -> (B, T, num_books, num_classes)
            xid = torch.einsum('btdh,dch->btdc', x, self.codebooks)
            # (B, T, num_books, num_classes) -> (B, T, num_books)
            xid = xid.max(dim=-1)[1]
        elif self.dist_fn == "l2":
            # (B, T, num_books, code_dim) -> (B, T, num_books, code_dim, num_classes)
            xid = x.unsqueeze(-1) - self.codebooks.transpose(1, 2).unsqueeze(0).unsqueeze(0)
            xid = xid.norm(dim=-2).argmin(dim=-1)
        else:
            raise ValueError(f"Unknown distance function {self.dist_fn}, must be one of {self.DIST_FN_LIST}")

        # xid2: (B, T, num_books) -> (B, T, num_books)
        xid2 = xid + self.num_classes * torch.arange(self.num_books, device=xid.device).unsqueeze(0).unsqueeze(0)
        # xid2: (B, T, num_books) -> (B*num_books, T)
        xid2 = xid2.transpose(1, 2).contiguous().view(-1, T)

        # get quantized vector (xq) of shape (B, T, code_dim, num_books)
        # codebook: (num_books, num_classes, code_dim) -> (num_books*num_classes, code_dim)
        xq = F.embedding(xid2.view(-1), self.codebooks.view(-1, self.code_dim)).view(
            B, T, self.code_dim, self.num_books
        )

        if not self.time_ahead:
            # (B, T, D) -> (B, D, T)
            xq = xq.transpose(1, 2)

        if self.num_books == 1 and self.squeeze_single:
            xq = xq.squeeze(-1)
            xid = xid.squeeze(-1)

        return xq, xid


class FSQVectorQuantizer(NeuralModule, Exportable):
    """Finite Scalar Quantization (FSQ) target tokenizer for NEST-style masked-token-prediction SSL.

    Drop-in replacement for ``RandomProjectionVectorQuantizer`` (BEST-RQ): it exposes the exact
    same forward contract (``input_signal -> (xq, xid)`` with the same shapes), so it works with
    ``MultiSoftmaxDecoder`` and ``MultiMLMLoss`` without any change. Differences from BEST-RQ:

      * Instead of nearest-neighbour lookup against a random codebook, each projected channel is
        independently rounded to a fixed grid of ``num_levels[d]`` integer levels (FSQ). The single
        target token is the mixed-radix combination of the per-channel digits, i.e. an integer in
        ``[0, prod(num_levels))``. There is no codebook table, so FSQ cannot collapse and uses
        (near) 100% of its vocabulary.
      * ``num_classes`` is therefore ``prod(num_levels)`` and there is a single codebook
        (``num_books == 1``). Set ``model.num_classes`` to ``prod(num_levels)`` and
        ``model.num_books: 1`` in the config so the decoder/loss vocabulary matches.

    The projection from input features to ``len(num_levels)`` dims mirrors BEST-RQ's design: a
    random, frozen ``nn.Linear`` (so the targets are a fixed function of the clean signal).

    FSQ math follows nemo.collections.tts.modules.audio_codec_modules.FiniteScalarQuantizer.

    Args:
        feat_in: input feature dimension (e.g. 80 mel bins).
        num_levels: list of quantization levels per code channel, e.g. [8, 8, 8, 4, 4] -> 8192.
        eps: scale shrink factor to keep tanh output off the rounding boundary.
        time_ahead: if True input is (B, T, D), otherwise (B, D, T).
        freeze: whether to freeze the (random) projection matrix.
        squeeze_single: if True, drop the trailing codebook dim (always a single book here).
        combine_time_steps: stack this many input frames before projecting (match encoder
            subsampling_factor for pre_conv masking, as in BEST-RQ).
    """

    def __init__(
        self,
        feat_in: int,
        num_levels: List[int],
        eps: float = 1e-3,
        time_ahead: bool = False,
        freeze: bool = True,
        squeeze_single: bool = False,
        combine_time_steps: int = 1,
    ):
        super().__init__()

        self.feat_in = feat_in
        self.num_levels_list = list(num_levels)
        self.code_dim = len(self.num_levels_list)
        self.num_books = 1
        self.num_classes = int(torch.prod(torch.tensor(self.num_levels_list)).item())
        self.eps = eps
        self.time_ahead = time_ahead
        self.squeeze_single = squeeze_single
        self.combine_time_steps = combine_time_steps

        # random, frozen projection: (B, T, feat_in*combine_time_steps) -> (B, T, code_dim)
        self.proj = nn.Linear(self.feat_in * combine_time_steps, self.code_dim, bias=False)
        torch.nn.init.xavier_normal_(self.proj.weight)

        # per-channel level vector and mixed-radix place values, shape (code_dim,)
        levels = torch.tensor(self.num_levels_list, dtype=torch.float32)
        dim_base_index = torch.cumprod(
            torch.tensor([1] + self.num_levels_list[:-1], dtype=torch.float32), dim=0
        )
        self.register_buffer("levels", levels)
        self.register_buffer("dim_base_index", dim_base_index)

        if freeze:
            self.freeze()

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        if self.time_ahead:
            return {"input_signal": NeuralType(('B', 'T', 'D'), SpectrogramType())}
        return {"input_signal": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        if self.time_ahead:
            if self.squeeze_single:
                return {
                    "xq": NeuralType(('B', 'T', 'D'), SpectrogramType()),
                    "xid": NeuralType(('B', 'T'), LabelsType()),
                }
            return {
                "xq": NeuralType(('B', 'T', 'D', 'H'), SpectrogramType()),
                "xid": NeuralType(('B', 'T', 'H'), LabelsType()),
            }
        if self.squeeze_single:
            return {
                "xq": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "xid": NeuralType(('B', 'T'), LabelsType()),
            }
        return {
            "xq": NeuralType(('B', 'D', 'T', 'H'), SpectrogramType()),
            "xid": NeuralType(('B', 'T', 'H'), LabelsType()),
        }

    def _quantize(self, x):
        """Apply FSQ to x of shape (B, T, code_dim). Returns (codes, indices).

        codes: dequantized values normalized to ~[-1, 1], shape (B, T, code_dim).
        indices: single token per frame in [0, num_classes), shape (B, T).
        """
        levels = self.levels  # (code_dim,)
        # --- compress each channel into its level range with a bounded tanh ---
        output_scale = (levels - 1) / 2 * (1 - self.eps)
        output_offset = torch.where(levels % 2 == 0, torch.full_like(levels, 0.5), torch.zeros_like(levels))
        input_shift = (output_offset / output_scale).tan()
        compressed = output_scale * (x + input_shift).tanh() - output_offset
        # --- round to nearest integer with straight-through estimator ---
        rounded = compressed + (torch.round(compressed) - compressed).detach()
        # --- normalize codes to ~[-1, 1] ---
        half = torch.div(levels, 2, rounding_mode='floor')
        codes = rounded / half
        # --- codes -> single mixed-radix index ---
        nonnegative = half * codes + half  # per-channel digit in [0, levels-1]
        indices = torch.sum(nonnegative * self.dim_base_index, dim=-1).to(torch.long)
        return codes, indices

    @typecheck()
    def forward(self, input_signal):
        """
        Args:
            input_signal: input features of shape (B, T, D) or (B, D, T).
        Returns:
            xq: dequantized features of shape (B, T, code_dim, 1) / (B, code_dim, T, 1)
                (trailing book dim dropped if squeeze_single).
            xid: target tokens of shape (B, T, 1) or (B, T) if squeeze_single.
        """
        if not self.time_ahead:
            # (B, D, T) -> (B, T, D)
            input_signal = input_signal.transpose(1, 2)

        B, T, _ = input_signal.size()

        if self.combine_time_steps > 1:
            input_signal = input_signal.contiguous().reshape(B, T // self.combine_time_steps, -1)
            T = T // self.combine_time_steps

        # (B, T, D) -> (B, T, code_dim)
        x = self.proj(input_signal)

        # FSQ: codes (B, T, code_dim), xid (B, T)
        codes, xid = self._quantize(x)

        # match BEST-RQ output layout: add trailing codebook dim
        xq = codes.unsqueeze(-1)  # (B, T, code_dim, 1)
        xid = xid.unsqueeze(-1)  # (B, T, 1)

        if not self.time_ahead:
            # (B, T, code_dim, 1) -> (B, code_dim, T, 1)
            xq = xq.transpose(1, 2)

        if self.squeeze_single:
            xq = xq.squeeze(-1)
            xid = xid.squeeze(-1)

        return xq, xid
