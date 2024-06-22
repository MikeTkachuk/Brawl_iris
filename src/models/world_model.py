from dataclasses import dataclass
from typing import Any, Optional, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from src.utils import init_weights, LossWithIntermediateLosses


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor


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
                    nn.Linear(self.config.embed_dim, self.config.embed_dim),
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
                continuous=None) -> WorldModelOutput:
        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        sequences = self.embedder(tokens, num_steps, prev_steps, continuous=continuous) + \
                    self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))

        x = self.transformer(sequences, past_keys_values)
        obs_per_head = [head(x, num_steps=num_steps, prev_steps=prev_steps) for head in self.obs_heads]

        logits_observations = rearrange(obs_per_head, "h b t e -> b (t h) e")
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)

        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:
        if "tokens" in batch:
            obs_tokens = batch["tokens"]
        else:
            with torch.no_grad():
                obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (B, L, K)
        act_tokens = batch['actions']
        act_continuous = batch['actions_continuous']
        tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))

        outputs = self(tokens, continuous=act_continuous)

        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'],
                                                                                           batch['ends'],
                                                                                           batch['mask_padding'],
                                                                                           )

        logits_observations = rearrange(outputs.logits_observations[:, :-self.config.n_gen_heads],
                                        'b t o -> (b t) o')
        loss_obs = F.cross_entropy(logits_observations, labels_observations)
        loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards,
                                       reduction="none")
        loss_rewards = (loss_rewards * (1 - (-loss_rewards).exp()).pow(2)).mean()
        loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends, reduction="none")
        loss_ends = (loss_ends * (1 - (-loss_ends).exp()).pow(2)).mean()

        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards,
                                          loss_ends=loss_ends)

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor,
                                   mask_padding: torch.BoolTensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)  # to be filled with -100 (default ignore index in F.cross_entropy)
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100),
                                        'b t k -> b (t k)')[:, self.config.n_gen_heads:]
        labels_rewards = self.encode_rewards(rewards).masked_fill(mask_fill,
                                                                  -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)
