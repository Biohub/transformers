# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""End-to-end CP runtime for ESMFold2's MSAEncoder.

``MSAEncoderCPWrapper`` is a drop-in replacement for the serial ``MSAEncoder``
(same plain-tensor in/out signature) that shards the MSA encoder's pair-space
work across the 2D CP grid, mirroring ``TrunkCPWrapper`` for the folding trunk.
Without it, the serial MSA encoder materialises the full L×L pair on every rank
(the dominant cost for large complexes) before the trunk even runs.
"""

from math import lcm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import Replicate, Shard, distribute_tensor


class MSAEncoderCPWrapper(nn.Module):
    """Drop-in replacement for ``MSAEncoder`` that runs distributed.

    The small per-residue embedding (``embed`` + ``project_inputs``) runs serial
    per rank (cheap), then m / pair / masks are zero-padded to a multiple of the
    CP shard factor, distributed as DTensors, processed by
    ``MSAEncoderDistributed``, and the gathered pair is sliced back.
    """

    def __init__(
        self, serial_encoder: nn.Module, dist_manager, comm: str = "gather"
    ) -> None:
        super().__init__()
        from transformers.models.esmfold2.distributed.manager import (
            DistributedManager,
        )
        from transformers.models.esmfold2.distributed.model.layers.msa_encoder import (
            MSAEncoderDistributed,
        )
        from transformers.models.esmfold2.modeling_esmfold2 import (
            MSAEncoder as SerialMSAEncoder,
        )

        if not isinstance(serial_encoder, SerialMSAEncoder):
            raise TypeError(
                f"expected MSAEncoder, got {type(serial_encoder).__name__}"
            )
        if not isinstance(dist_manager, DistributedManager):
            raise TypeError(
                f"expected DistributedManager, got {type(dist_manager).__name__}"
            )

        # The distributed path ignores the serial chunk-size knob.
        serial_encoder.set_chunk_size(None)

        # The embedding projections stay serial (tiny); reference them directly.
        self.embed = serial_encoder.embed
        self.project_inputs = serial_encoder.project_inputs
        self.dist_encoder = MSAEncoderDistributed(serial_encoder, dist_manager, comm=comm)

        self.dist_manager = dist_manager
        self.device_mesh = dist_manager.device_mesh_subgroups
        # device_mesh is (dp, cp_axis_0, cp_axis_1)
        self.cp_axis_0 = self.device_mesh.size(1)
        self.cp_axis_1 = self.device_mesh.size(2)
        self.shard_factor = lcm(self.cp_axis_0, self.cp_axis_1)

    def set_chunk_size(self, _chunk_size: int | None) -> None:
        return

    def set_kernel_backend(self, _backend: str | None) -> None:
        return

    def forward(
        self,
        x_pair: torch.Tensor,
        x_inputs: torch.Tensor,
        msa_oh: torch.Tensor,
        has_deletion: torch.Tensor,
        deletion_value: torch.Tensor,
        msa_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Serial embedding (matches MSAEncoder.forward), per rank on full tensors.
        m_feat = torch.cat(
            [msa_oh, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1
        )
        m = self.embed(m_feat) + self.project_inputs(x_inputs).unsqueeze(2)

        n = x_pair.shape[1]
        pad = (self.shard_factor - n % self.shard_factor) % self.shard_factor
        if pad:
            m = F.pad(m, (0, 0, 0, 0, 0, pad))  # pad L (dim 1)
            x_pair = F.pad(x_pair, (0, 0, 0, pad, 0, pad))  # pad both L dims
            msa_attention_mask = F.pad(msa_attention_mask, (0, 0, 0, pad))  # pad L

        # Build the pair attention mask from the (padded) token mask; padded
        # rows/cols are zero so they contribute nothing.
        tok_mask = msa_attention_mask[:, :, 0].bool()
        pair_mask = (tok_mask.unsqueeze(2) & tok_mask.unsqueeze(1)).to(m.dtype)

        mesh = self.device_mesh
        m_dt = distribute_tensor(
            m.contiguous(), mesh, [Shard(0), Shard(1), Replicate()]
        )
        pair_dt = distribute_tensor(
            x_pair.contiguous(), mesh, [Shard(0), Shard(1), Shard(2)]
        )
        msa_mask_dt = distribute_tensor(
            msa_attention_mask.to(m.dtype).contiguous(),
            mesh,
            [Shard(0), Shard(1), Replicate()],
        )
        pair_mask_dt = distribute_tensor(
            pair_mask.contiguous(), mesh, [Shard(0), Shard(1), Shard(2)]
        )

        out_dt = self.dist_encoder(m_dt, pair_dt, msa_mask_dt, pair_mask_dt)
        out = out_dt.full_tensor()
        if pad:
            out = out[:, :n, :n, :]
        return out


def wrap_model_with_cp_msa_encoder(
    model: nn.Module, dist_manager, comm: str = "gather"
) -> list[str]:
    """Replace every ``MSAEncoder`` submodule with ``MSAEncoderCPWrapper``.

    Walks ``model.named_modules()`` and rebinds each attribute that points at a
    serial ``MSAEncoder``. Returns the list of replaced submodule paths. Models
    without an MSA encoder (``model.msa_encoder is None``) are left untouched.

    ``comm`` selects the MSA pair-weighted-averaging communication strategy:
    ``"gather"`` (default, bit-exact all-gather) or ``"ring"`` (boltz-style
    online-softmax ring, for large grids / deep MSAs).
    """
    mesh = dist_manager.device_mesh_subgroups
    cp0, cp1 = mesh.size(1), mesh.size(2)
    if cp0 != cp1:
        raise ValueError(
            f"CP grid must be square (cp_axis_0 == cp_axis_1), got {cp0}×{cp1}"
        )

    from transformers.models.esmfold2.modeling_esmfold2 import (
        MSAEncoder as SerialMSAEncoder,
    )

    targets: list[tuple[str, nn.Module, str, nn.Module]] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if child is None:
                continue
            if isinstance(child, SerialMSAEncoder):
                full = f"{parent_name}.{child_name}" if parent_name else child_name
                targets.append((full, parent, child_name, child))

    replaced: list[str] = []
    for full, parent, child_name, child in targets:
        wrapped = MSAEncoderCPWrapper(child, dist_manager, comm=comm).to(
            device=dist_manager.device, dtype=next(child.parameters()).dtype
        )
        setattr(parent, child_name, wrapped)
        replaced.append(full)
    return replaced


def wrap_model_with_cp(
    model: nn.Module,
    dist_manager,
    comm: str = "gather",
    bf16: bool = True,
    offload_esmc: bool = True,
) -> list[str]:
    """Wrap both the folding trunk(s) and the MSA encoder for 2D CP.

    Convenience over calling ``wrap_model_with_cp_trunks`` and
    ``wrap_model_with_cp_msa_encoder`` separately. Returns the combined list of
    replaced submodule paths.

    ``comm`` selects the MSA pair-weighted-averaging communication strategy:
    ``"gather"`` (default, bit-exact all-gather) or ``"ring"`` (boltz-style
    online-softmax ring). It affects only the MSA encoder; the folding trunk
    and the OuterProductMean (whose contracted MSA-depth axis is replicated)
    are unaffected.

    ``bf16`` (default True): run the distributed trunk in bf16 — quality-neutral,
    ~1.3-1.6x faster, lower peak VRAM (see ``wrap_model_with_cp_trunks``).

    ``offload_esmc`` (default True): offload the ESM-C LM (~12 GB) to CPU after
    its one-shot use, freeing it for the trunk/diffusion. Sets
    ``model._offload_esmc``. Validated with bf16: 2.14x end-to-end + 18.5 GB lower
    peak vs fp32, pLDDT unchanged. Set False for high-throughput tiny-fold or
    concurrent-fold use (the per-fold ~12 GB CPU<->GPU transfer isn't worth it).
    """
    from transformers.models.esmfold2.distributed.utils import (
        wrap_model_with_cp_trunks,
    )

    model._offload_esmc = offload_esmc
    replaced = wrap_model_with_cp_trunks(model, dist_manager, bf16=bf16)
    replaced += wrap_model_with_cp_msa_encoder(model, dist_manager, comm=comm)
    return replaced
