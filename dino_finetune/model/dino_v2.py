import math
from typing import Optional

import torch
import torch.nn as nn

from .lora import LoRA
from .linear_decoder import LinearClassifier


class DINOV2EncoderLoRA(nn.Module):
    def __init__(
        self,
        encoder,
        r: int = 4,
        n: Optional[int] = None,
        n_classes: int = 1000,
        emb_dim: int = 1024,
        img_dim: tuple[int, int] = (520, 520),
        use_lora: bool = False,
    ):
        super().__init__()
        assert img_dim[0] % encoder.patch_size == 0, "Wrong input shape for patches"
        assert r > 0

        self.n = n
        self.emb_dim = emb_dim
        self.img_dim = img_dim
        self.use_lora = use_lora

        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.lora_layers = list(range(len(self.encoder.blocks)))

        # Decoder
        # Patch size is given by (490/14)**2 = 35 * 35
        self.decoder = LinearClassifier(
            emb_dim,
            patch_h=int(img_dim[0] / encoder.patch_size),
            patch_w=int(img_dim[1] / encoder.patch_size),
            n_classes=n_classes,
        )

        # Add LoRA layers to the encoder
        if self.use_lora:
            self.w_As = []
            self.w_Bs = []

            for i, block in enumerate(self.encoder.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

                self.w_As.extend([w_a_linear_q, w_a_linear_v])
                self.w_Bs.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            self.reset_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: torch.Tensor, return_patches: bool = False):
        feature = self.encoder.forward_features(x)

        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = feature["x_norm_patchtokens"]
        logits = self.decoder(patch_embeddings)
        logits = nn.functional.interpolate(
            logits,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        if return_patches:
            return logits, patch_embeddings
        return logits
