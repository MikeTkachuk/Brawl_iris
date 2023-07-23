"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Any, Tuple

from einops import rearrange
import torch
import torch.nn as nn

from src.dataset import Batch
from .lpips import LPIPS
from .nets import Encoder, Decoder
from src.utils import LossWithIntermediateLosses


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor


class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder,
                 with_lpips: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.pre_quant_conv = torch.nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.lpips = LPIPS().eval() if with_lpips else None

        self._token_histogram = None

    def __repr__(self) -> str:
        return "tokenizer"

    @property
    def normed_embedding(self):
        return self.embedding.weight / (1E-8 + torch.norm(self.embedding.weight, dim=1, keepdim=True))

    def get_param_groups(self, weight_decay=0.01):
        wd_parameters = ['embedding.weight']
        optim_groups = [
            {"params": [self.get_parameter(param) for param in wd_parameters], "weight_decay": weight_decay},
            {"params": [param for param_name, param in self.named_parameters()
                        if param_name not in wd_parameters], "weight_decay": 0.0},
        ]
        return optim_groups

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[
        torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        # decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        normed_z = outputs.z / (1E-8 + torch.norm(outputs.z, dim=1, keepdim=True))
        norm_z_quant = 1E-8 + torch.norm(outputs.z_quantized, dim=1, keepdim=True)
        normed_z_quant = outputs.z_quantized / norm_z_quant
        decoder_input = (normed_z + (normed_z_quant - normed_z).detach()) * norm_z_quant
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(self, batch: Batch, commitment_start_after_epochs: int, epoch: int, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 0.25 if commitment_start_after_epochs < epoch else 0.0
        # commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()
        commitment_loss = (1 + beta) - torch.cosine_similarity(z.detach(), z_quantized, dim=1).mean() - \
                           beta * torch.cosine_similarity(z, z_quantized.detach(), dim=1).mean()

        embedding_cosines = torch.tril(self.normed_embedding @ self.normed_embedding.T, diagonal=-1)
        tolerance_mask = embedding_cosines > 0.3
        if torch.count_nonzero(tolerance_mask) > 0:
            orthogonality_loss = beta * embedding_cosines[tolerance_mask].mean()
        else:
            orthogonality_loss = torch.tensor(0.0)

        reconstruction_loss = torch.abs(observations - reconstructions).mean()
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))

        return LossWithIntermediateLosses(commitment_loss=commitment_loss,
                                          orthogonality_loss=orthogonality_loss,
                                          reconstruction_loss=reconstruction_loss,
                                          perceptual_loss=perceptual_loss)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        z = self.pre_quant_conv(z)
        b, e, h, w = z.shape
        z_flattened = rearrange(z, 'b e h w -> (b h w) e')
        dist_to_embeddings = (z_flattened / (1E-8 + torch.norm(z_flattened, dim=1, keepdim=True))) @ (
                self.embedding.weight / (1E-8 + torch.norm(self.embedding.weight, dim=1, keepdim=True))).T
        # dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        tokens = dist_to_embeddings.argmax(dim=-1)
        z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()

        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)

        if self.training and self._token_histogram is not None:
            for t in tokens.reshape(-1):
                self._token_histogram[t] += 1

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        z_q = self.post_quant_conv(z_q)
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False,
                      should_postprocess: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)
