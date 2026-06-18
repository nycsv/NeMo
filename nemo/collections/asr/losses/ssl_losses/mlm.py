# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.nn.functional as F
from torch import nn

from nemo.core import Loss, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType, SpectrogramType

__all__ = ["MLMLoss", "MultiMLMLoss", "MaskedConsistencyLoss"]


class MLMLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for Contrastive."""
        return {
            "spec_masks": NeuralType(("B", "D", "T"), SpectrogramType(), optional=True),
            "decoder_outputs": NeuralType(("B", "T", "D"), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "decoder_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "target_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "masks": NeuralType(("B", "D", "T"), SpectrogramType(), optional=True),
        }

    @property
    def output_types(self):
        """Output types definitions for Contrastive.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    @property
    def needs_labels(self):
        return True

    def __init__(
        self,
        combine_time_steps: int = 1,
        mask_threshold: float = 0.8,
    ):
        super().__init__()
        self.nll_loss = nn.NLLLoss()
        self.combine_time_steps = combine_time_steps
        self.mask_threshold = mask_threshold

    @typecheck()
    def forward(
        self, decoder_outputs, targets, decoder_lengths=None, target_lengths=None, spec_masks=None, masks=None
    ):

        if masks is None:
            masks = spec_masks

        if masks is None:
            masks = torch.ones_like(decoder_outputs, dtype=torch.bool)
        else:
            # B,D,T -> B,T,D
            masks = masks.transpose(1, 2)

            masks = masks.reshape(masks.shape[0], masks.shape[1] // self.combine_time_steps, -1)
            masks = masks.mean(-1) > self.mask_threshold

        out_masked_only = decoder_outputs[masks]
        targets = F.pad(targets, (0, masks.shape[-1] - targets.shape[-1]))
        targets_masked_only = targets[masks]

        loss = self.nll_loss(out_masked_only, targets_masked_only)
        loss = torch.mean(loss)

        return loss


class MultiMLMLoss(Loss):
    """
    Masked language model loss for multiple decoders, where cross-entropy loss is applied separately on each decoder.
    This loss can be used with `nemo.collections.asr.modules.ssl_modules.MultiSoftmaxDecoder` to train a model with multiple targets per frame.
    Reference: https://arxiv.org/abs/2202.01855
    """

    @property
    def input_types(self):
        if self.squeeze_single and self.num_decoders == 1:
            decoder_outputs = NeuralType(("B", "T", "C"), LogprobsType())
            targets = NeuralType(('B', 'T'), LabelsType())
        else:
            decoder_outputs = NeuralType(("B", "T", "C", "H"), LogprobsType())
            targets = NeuralType(("B", "T", "H"), LabelsType())
        return {
            "masks": NeuralType(("B", "D", "T"), SpectrogramType()),
            "decoder_outputs": decoder_outputs,
            "targets": targets,
            "decoder_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "target_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        combine_time_steps: int = 1,
        mask_threshold: float = 0.8,
        num_decoders: int = 1,
        squeeze_single: bool = False,
    ):
        super().__init__()
        self.num_decoders = num_decoders
        self.squeeze_single = squeeze_single
        self.mlm_loss = MLMLoss(combine_time_steps, mask_threshold)

    @typecheck()
    def forward(self, masks, decoder_outputs, targets, decoder_lengths=None, target_lengths=None):
        if self.squeeze_single and self.num_decoders == 1:
            return self.mlm_loss(
                spec_masks=masks,
                decoder_outputs=decoder_outputs,
                targets=targets,
                decoder_lengths=decoder_lengths,
                target_lengths=target_lengths,
            )
        loss = 0.0
        for i in range(self.num_decoders):
            loss += self.mlm_loss(
                spec_masks=masks,
                decoder_outputs=decoder_outputs[:, :, :, i],
                targets=targets[:, :, i],
                decoder_lengths=decoder_lengths,
                target_lengths=target_lengths,
            )
        return loss / self.num_decoders


class MaskedConsistencyLoss(Loss):
    """Dual-mode consistency loss for streaming SSL.

    Distills a full/large-context "teacher" pass into a limited-context streaming "student" pass
    by minimizing the KL divergence between their per-frame token distributions on the *masked*
    frames only (the same frames the MLM loss is computed on). The teacher is treated as a fixed
    target (stop-gradient), so the gradient only updates the streaming path:

        L_consistency = KL( softmax(teacher) || softmax(student) )   over masked frames.

    Both ``student_outputs`` and ``teacher_outputs`` are expected to be log-probabilities, exactly
    as produced by ``MultiSoftmaxDecoder`` (which applies log_softmax). The masking convention
    matches ``MLMLoss``: the (B, D, T) spectrogram-level mask is pooled by ``combine_time_steps``
    and a frame counts as masked when its mean mask value exceeds ``mask_threshold``.
    """

    @property
    def input_types(self):
        if self.squeeze_single and self.num_decoders == 1:
            outputs = NeuralType(("B", "T", "C"), LogprobsType())
        else:
            outputs = NeuralType(("B", "T", "C", "H"), LogprobsType())
        return {
            "masks": NeuralType(("B", "D", "T"), SpectrogramType()),
            "student_outputs": outputs,
            "teacher_outputs": outputs,
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        combine_time_steps: int = 1,
        mask_threshold: float = 0.8,
        num_decoders: int = 1,
        squeeze_single: bool = False,
    ):
        super().__init__()
        self.combine_time_steps = combine_time_steps
        self.mask_threshold = mask_threshold
        self.num_decoders = num_decoders
        self.squeeze_single = squeeze_single

    def _masked_kl(self, masks, student_lp, teacher_lp):
        # student_lp / teacher_lp: (B, T, C); masks: (B, D, T) spectrogram-level mask
        masks = masks.transpose(1, 2)  # (B, T_spec, D)
        masks = masks.reshape(masks.shape[0], masks.shape[1] // self.combine_time_steps, -1)
        masks = masks.mean(-1) > self.mask_threshold  # (B, T)
        # guard against off-by-one between mask time steps and decoder time steps
        t = min(masks.shape[1], student_lp.shape[1])
        masks = masks[:, :t]
        student_sel = student_lp[:, :t][masks]  # (N, C)
        teacher_sel = teacher_lp[:, :t][masks].detach()  # (N, C), stop-gradient teacher
        if student_sel.numel() == 0:
            return student_lp.new_zeros(())
        # KL(teacher || student) with both args as log-probabilities
        return F.kl_div(student_sel, teacher_sel, reduction="batchmean", log_target=True)

    @typecheck()
    def forward(self, masks, student_outputs, teacher_outputs):
        if self.squeeze_single and self.num_decoders == 1:
            return self._masked_kl(masks, student_outputs, teacher_outputs)
        loss = 0.0
        for i in range(self.num_decoders):
            loss += self._masked_kl(masks, student_outputs[:, :, :, i], teacher_outputs[:, :, :, i])
        return loss / self.num_decoders
