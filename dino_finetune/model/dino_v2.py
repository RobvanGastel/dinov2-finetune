import math
from typing import Optional

import torch
import torch.nn as nn

from .lora import LoRA
from .linear_decoder import LinearClassifier
from .uper_decoder import UperDecoder


class DINOV2EncoderLoRA(nn.Module):
    def __init__(
        self,
        encoder,
        r: int = 4,
        n: Optional[int] = None,
        emb_dim: int = 1024,
        n_classes: int = 1000,
        use_lora: bool = False,
        use_uper: bool = False,
        img_dim: tuple[int, int] = (520, 520),
    ):
        super().__init__()
        assert img_dim[0] % encoder.patch_size == 0, "Wrong input shape for patches"
        assert r > 0

        self.n = n
        self.emb_dim = emb_dim
        self.img_dim = img_dim
        self.use_lora = use_lora

        # Number of previous layers to use as input
        self.uper_dim = 4
        self.use_uper = use_uper

        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.lora_layers = list(range(len(self.encoder.blocks)))

        # Decoder
        # Patch size is given by (490/14)**2 = 35 * 35
        if self.use_uper:
            self.decoder = UperDecoder(
                [emb_dim] * self.uper_dim,
                out_channels=128,
                patch_h=int(img_dim[0] / encoder.patch_size),
                patch_w=int(img_dim[1] / encoder.patch_size),
                n_classes=n_classes,
            )
        else:
            self.decoder = LinearClassifier(
                emb_dim,
                patch_h=int(img_dim[0] / encoder.patch_size),
                patch_w=int(img_dim[1] / encoder.patch_size),
                n_classes=n_classes,
            )

        # Add LoRA layers to the encoder
        if self.use_lora:
            self.w_a = []
            self.w_b = []

            for i, block in enumerate(self.encoder.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # If the uperNet decoder is used, we take the n last layers for
        # our decoder to get a better segmentation result.
        if self.use_uper:
            # Potentially even better to take a different depths
            feature = self.encoder.get_intermediate_layers(
                x, n=self.uper_dim, reshape=True
            )
            logits = self.decoder(feature)

        else:
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
        return logits

    def save_parameters(self, filename: str) -> None:
        """Save the LoRA weights and decoder weights to a .pt file

        Args:
            filename (str): Filename of the weights
        """
        w_a, w_b = {}, {}
        if self.use_lora:
            w_a = {f"w_a_{i:03d}": self.w_a[i].weight for i in range(len(self.w_a))}
            w_b = {f"w_b_{i:03d}": self.w_b[i].weight for i in range(len(self.w_a))}

        decoder_weights = self.decoder.state_dict()
        torch.save({**w_a, **w_b, **decoder_weights}, filename)

    def load_parameters(self, filename: str) -> None:
        """Load the LoRA and decoder weights from a file

        Args:
            filename (str): File name of the weights
        """
        state_dict = torch.load(filename)

        # Load the LoRA parameters
        if self.use_lora:
            for i, w_A_linear in enumerate(self.w_a):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_A_linear.weight = nn.Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_b):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_B_linear.weight = nn.Parameter(saved_tensor)

        # Load decoder parameters
        decoder_head_dict = self.decoder.state_dict()
        decoder_head_keys = [k for k in decoder_head_dict.keys()]
        decoder_state_dict = {k: state_dict[k] for k in decoder_head_keys}

        decoder_head_dict.update(decoder_state_dict)
        self.decoder.load_state_dict(decoder_head_dict)
