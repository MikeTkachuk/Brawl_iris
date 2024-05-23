"""
Credits to https://github.com/CompVis/taming-transformers
"""
import time
from dataclasses import dataclass
from typing import Any, Tuple

from einops import rearrange
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import Batch
from .lpips import LPIPS
from .nets import Encoder, Decoder, Normalize
from .adversarial import AdversarialLoss
from src.utils import LossWithIntermediateLosses

from sklearn.cluster import BisectingKMeans


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor


class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder,
                 with_lpips: bool = True, loss_weights=(1.0, 1.0, 1.0, 1.0, 1.0), gan_loss=False) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.pre_quant_norm = Normalize(encoder.config.z_channels, num_groups=32)
        self.pre_quant_conv = torch.nn.Conv2d(encoder.config.z_channels,
                                              embed_dim, 1, bias=False)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.lpips = LPIPS().eval() if with_lpips else None
        self.loss_weights = loss_weights

        self.ad_loss_enabled = gan_loss
        if self.ad_loss_enabled:
            self.ad_loss = AdversarialLoss()
        self.param_groups = {
            "gen": [],
            "discr": []
        }
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "ad_loss" in n:
                self.param_groups["discr"].append(p)
            else:
                self.param_groups["gen"].append(p)

    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x: torch.Tensor,
                should_preprocess: bool = False,
                should_postprocess: bool = False) -> Tuple[TokenizerEncoderOutput, torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs, reconstructions

    def compute_loss(self, batch: Batch,
                     tokens_placeholder=None,
                     pixel_weights=None,
                     **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        enc_output, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)
        z, z_quantized = enc_output.z, enc_output.z_quantized
        if tokens_placeholder is not None:
            tokens_placeholder[0] = enc_output.tokens

        beta = 0.25  # simplify z optimization. codebook will be updated with kmeans
        commitment_loss = (1 - beta) * (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()
        if pixel_weights is not None:
            reconstruction_loss = (pixel_weights * torch.square(observations - reconstructions)).mean()
        else:
            reconstruction_loss = torch.square(observations - reconstructions).mean()
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))
        if self.ad_loss_enabled:
            ad_logits = self.ad_loss(torch.cat([observations, reconstructions]), blur=True).flatten()
            discr_labels = torch.zeros((observations.size(0)*2,), dtype=torch.float, device=ad_logits.device)
            discr_labels[:observations.size(0)] = 1
            discr_loss = nn.functional.binary_cross_entropy_with_logits(ad_logits, discr_labels)
            gen_loss = nn.functional.binary_cross_entropy_with_logits(ad_logits[observations.size(0):],
                                                   torch.ones_like(discr_labels)[:observations.size(0)])
        else:
            discr_loss = 0
            gen_loss = 0

        return LossWithIntermediateLosses(commitment_loss=self.loss_weights[0]*commitment_loss,
                                          reconstruction_loss=self.loss_weights[1]*reconstruction_loss,
                                          perceptual_loss=self.loss_weights[2]*perceptual_loss,
                                          discr_loss=self.loss_weights[3]*discr_loss,
                                          gen_loss=self.loss_weights[4]*gen_loss)

    def do_backward(self, loss):
        if self.ad_loss_enabled:
            loss.backward(inputs=self.param_groups["discr"], retain_graph=True)
        loss.backward(inputs=self.param_groups["gen"])

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        z = self.pre_quant_norm(z)
        z = self.pre_quant_conv(z)
        b, e, h, w = z.shape
        z_flattened = rearrange(z, 'b e h w -> (b h w) e')
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                             torch.sum(self.embedding.weight ** 2, dim=1) - \
                             2 * torch.matmul(z_flattened, self.embedding.weight.t())

        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()

        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)

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
                      should_postprocess: bool = False, should_quantize: bool = True) -> torch.Tensor:
        if should_quantize:
            z = self.encode(x, should_preprocess).z_quantized
        else:
            z = self.encode(x, should_preprocess).z
        return self.decode(z, should_postprocess)

    @torch.no_grad()
    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    @torch.no_grad()
    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)

    @torch.no_grad()
    def init_embedding_kmeans(self, data, num_batches=64):
        embeddings = []
        device = self.parameters().__next__().device
        for i, sample in tqdm(enumerate(data), desc="Preparing kmeans embeddings:", total=num_batches):
            if i >= num_batches:
                break
            batch, _ = sample
            emb = self.encode(batch["observations"].to(device)).z
            embeddings.append(emb)
        embeddings = rearrange(embeddings, '... c h w -> (... h w) c')
        print("Starting kmeans fit.")
        kmeans = BisectingKMeans(n_clusters=self.vocab_size, verbose=0, n_init=1, init="random")
        kmeans.fit(embeddings.cpu().numpy())
        self.embedding.weight.data.copy_(torch.from_numpy(kmeans.cluster_centers_))

