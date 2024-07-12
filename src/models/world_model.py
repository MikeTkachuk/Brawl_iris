from dataclasses import dataclass
from typing import Any, Optional, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from src.utils import init_weights, LossWithIntermediateLosses, compute_masked_lambda_returns


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    block_values: torch.FloatTensor
    gen_tokens: torch.LongTensor


class ObsHeadModule(nn.Module):
    def __init__(self, n_extra=0, emb_size=512):
        super().__init__()
        self.n_extra = n_extra
        self.emb_size = emb_size
        self.reducers = nn.ModuleList()
        self.reduced_emb_size = self.emb_size // 8
        for i in range(self.n_extra):
            self.reducers.append(nn.Linear(self.emb_size, self.reduced_emb_size))
        self.merger = nn.Linear(self.emb_size + self.reduced_emb_size * self.n_extra, self.emb_size)

    def forward(self, x: torch.Tensor):
        """

        :param x: expects transformer sequence at position 0
        :return:
        """
        b, t, n, e = x.size()
        assert n == self.n_extra + 1, f"Expecting {self.n_extra} extra embeddings in dim 2. Got {n} total."
        reduced = [self.reducers[i](x[..., i + 1, :]) for i in range(self.n_extra)]
        reduced = rearrange(reduced, "n b t e -> b t (n e)") if reduced else x[..., 1:, :].flatten(-2)
        merged = torch.cat([x[..., 0, :], reduced], dim=-1)
        return self.merger(merged)


class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig,
                 act_continuous_size: int, reward_map: list, reward_divisor: int) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config
        self.transformer = Transformer(config)
        self.act_continuous_size = act_continuous_size
        self.rewards_map = dict(zip(reward_map, range(len(reward_map))))
        self.reward_list = torch.tensor(reward_map)
        self.reward_divisor = reward_divisor

        act_tokens_pattern = torch.zeros(config.tokens_per_block)
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern
        nd_last_obs_pattern = torch.zeros(config.tokens_per_block)
        nd_last_obs_pattern[-3] = 1  #

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList(
                [nn.Embedding(act_vocab_size, config.embed_dim), nn.Embedding(obs_vocab_size, config.embed_dim)]
            ),
            continuous_size=self.act_continuous_size,
            action_table_id=0
        )
        self.obs_heads = self._get_obs_heads()

        self.head_rewards = Head(
            max_blocks=config.max_blocks,
            block_mask=nd_last_obs_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, len(self.reward_list))
            )
        )

        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=nd_last_obs_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2)
            )
        )

        self.head_values = Head(
            max_blocks=config.max_blocks,
            block_mask=nd_last_obs_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 1)
            )
        )

        self.apply(init_weights)

    def __repr__(self) -> str:
        return "world_model"

    def _get_obs_heads(self):
        """
        Get a list of heads generating n tokens at once
        :return:
        """
        num_chunks = (self.config.tokens_per_block - 1) // self.config.n_gen_heads
        assert self.config.tokens_per_block - 1 == self.config.n_gen_heads * num_chunks
        mask = [1] + (self.config.n_gen_heads - 1) * [0]
        mask = num_chunks * mask
        mask = torch.tensor(mask[1:] + [0, 1]).int()  # shift by one and add "0" for action token prediction
        heads = []
        for i in range(self.config.n_gen_heads):
            head = Head(
                max_blocks=self.config.max_blocks,
                block_mask=mask,
                head_module=nn.Sequential(
                    ObsHeadModule(emb_size=self.config.embed_dim, n_extra=i),
                    nn.ReLU(),
                    nn.Linear(self.config.embed_dim, self.obs_vocab_size)
                )
            )
            heads.append(head)
        return torch.nn.ModuleList(heads)

    def encode_rewards(self, rewards):

        def _mapper(x):
            x = round(x * self.reward_divisor)
            return self.rewards_map[x]

        return rewards.to("cpu").apply_(_mapper).to(rewards.device)

    def decode_rewards(self, rewards):
        return self.reward_list[rewards.to("cpu")].to(rewards.device) / self.reward_divisor

    def forward(self, tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None,
                continuous=None, temperature=1.0) -> WorldModelOutput:
        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        sequences = self.embedder(tokens, num_steps, prev_steps, continuous=continuous) + \
                    self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))

        x = self.transformer(sequences, past_keys_values)
        head_slice = self.obs_heads[0].compute_slice(num_steps=num_steps, prev_steps=prev_steps)
        obs_slice = self.embedder.slicers[1].compute_slice(num_steps=num_steps, prev_steps=prev_steps)
        extra_seq = rearrange(sequences[:, obs_slice], "b (n nh) e -> b n nh e", nh=self.config.n_gen_heads)
        extra_seq = extra_seq[:, 1:]  # extra starts at #n_gen_heads
        gen_tokens = []
        obs_per_head = []
        for i, head in enumerate(self.obs_heads):
            x_obs = nn.functional.pad(x.unsqueeze(-2), pad=[0, 0, 0, i])
            x_obs[:, head_slice[:-1], 1:] = extra_seq[:, :len(head_slice)-1, :i]
            if gen_tokens and head_slice.size(0):
                gen_seq = self.embedder(torch.stack(gen_tokens, dim=1), len(gen_tokens), prev_steps + i,) + \
                          self.pos_emb((prev_steps + torch.arange(i, device=tokens.device)) % self.config.max_tokens)
                x_obs[:, head_slice[-1], 1:] = gen_seq

            logits = head(x_obs, num_steps=num_steps, prev_steps=prev_steps)
            obs_per_head.append(logits)
            if head_slice.size(0):
                token_dist = Categorical(logits=logits[:, -1]/temperature)
                token = token_dist.sample()
            else:
                token = torch.zeros((logits.size(0),), dtype=torch.long, device=logits.device)
            gen_tokens.append(token)

        # obs_per_head = [head(x, num_steps=num_steps, prev_steps=prev_steps) for head in self.obs_heads]

        logits_observations = rearrange(obs_per_head, "h b t e -> b (t h) e")
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)
        block_values = self.head_values(x, num_steps=num_steps, prev_steps=prev_steps)[..., 0]
        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends,
                                block_values, torch.stack(gen_tokens, dim=1))

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:
        if "tokens" in batch:
            obs_tokens = batch["tokens"]
        else:
            with torch.no_grad():
                obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (B, L, K)
        act_tokens = batch['actions']
        act_continuous = batch['actions_continuous']
        tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))

        outputs: WorldModelOutput = self(tokens, continuous=act_continuous)

        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'],
                                                                                           batch['ends'],
                                                                                           batch['mask_padding'],
                                                                                           )
        obs_slice = slice(self.config.tokens_per_block - self.config.n_gen_heads - 1, -self.config.n_gen_heads)
        logits_observations = rearrange(outputs.logits_observations[:, obs_slice],
                                        'b t o -> (b t) o')
        loss_obs = F.cross_entropy(logits_observations, labels_observations)
        loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards,
                                       reduction="none")
        loss_rewards = (loss_rewards * (1 - (-loss_rewards).exp()).pow(2)).mean()
        loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends, reduction="none")
        loss_ends = (loss_ends * (1 - (-loss_ends).exp()).pow(2)).mean()
        with torch.no_grad():
            lambda_returns = compute_masked_lambda_returns(batch['rewards'],
                                                           outputs.block_values.detach(),
                                                           batch['ends'],
                                                           batch['mask_padding'],
                                                           kwargs.get("gamma", 0.995),
                                                           kwargs.get("lambda_", 0.95)
                                                           )
        loss_values = torch.square(outputs.block_values - lambda_returns)
        loss_values = torch.masked_select(loss_values, batch['mask_padding']).mean()

        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards,
                                          loss_ends=loss_ends, loss_values=loss_values)

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor,
                                   mask_padding: torch.BoolTensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)  # to be filled with -100 (default ignore index in F.cross_entropy)
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100),
                                        'b t k -> b (t k)')[:, self.config.tokens_per_block - 1:]
        labels_rewards = self.encode_rewards(rewards).masked_fill(mask_fill,
                                                                  -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)
