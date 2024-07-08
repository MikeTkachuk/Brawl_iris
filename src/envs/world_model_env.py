import random
from typing import List, Optional, Union
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision

from src.utils import extract_state_dict


class WorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env=None) -> None:

        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens = None, None, None

        self.env = env

        self.temperature = 1.0

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]
        return self.reset_from_initial_observations(obs)

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens    # (B, C, H, W) -> (B, K)
        _, num_observations_tokens = obs_tokens.shape
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens

        _ = self.refresh_keys_values_with_initial_obs_tokens(obs_tokens)
        self.obs_tokens = obs_tokens

        return self.decode_obs_tokens()

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        n, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
        outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm)
        return outputs_wm.output_sequence  # (B, K, E)

    @torch.no_grad()
    def step(self, action: Union[int, torch.LongTensor],
             continuous=None):
        """

        :param action:
        :param continuous:
        :return:
        """
        max_context = self.world_model.config.max_tokens
        assert self.keys_values_wm is not None and self.num_observations_tokens is not None
        num_heads = self.world_model.config.n_gen_heads
        assert self.num_observations_tokens % num_heads == 0
        num_passes = self.num_observations_tokens // num_heads

        output_sequence, obs_tokens, obs_probas = [], [], []

        if self.keys_values_wm.size + num_passes > max_context:
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

        tokens = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        tokens = tokens.reshape(-1, 1).to(self.device)  # (B, 1)
        if continuous is not None:
            continuous = continuous.clone().detach() if isinstance(continuous, torch.Tensor) else torch.tensor(continuous,
                                                                                                  dtype=torch.float)
            continuous = continuous.reshape(-1, 1, self.world_model.act_continuous_size).to(self.device)  # (B, 1, #act_continuous)

        for k in range(num_passes):  # assumption that there is only one action token.

            outputs_wm = self.world_model(tokens, past_keys_values=self.keys_values_wm, continuous=continuous)
            output_sequence.append(outputs_wm.output_sequence)

            token_dist = Categorical(logits=outputs_wm.logits_observations/self.temperature)
            tokens = token_dist.sample()
            obs_tokens.append(tokens)
            obs_probas.append(token_dist.log_prob(tokens))

        # add last tokens to context
        outputs_wm = self.world_model(tokens, past_keys_values=self.keys_values_wm, continuous=continuous)
        reward = Categorical(logits=outputs_wm.logits_rewards / self.temperature).sample()
        reward = self.world_model.decode_rewards(reward).float().cpu().numpy().reshape(-1)  # (B,)
        done = Categorical(logits=outputs_wm.logits_ends / self.temperature).sample().cpu().numpy().astype(
            bool).reshape(-1)  # (B,)

        self.obs_tokens = torch.cat(obs_tokens, dim=1)        # (B, K)
        self.obs_probas = torch.cat(obs_probas, dim=1)
        obs = self.decode_obs_tokens()
        return obs, reward, done, {"value": outputs_wm.block_values.flatten().cpu().numpy()}

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self) -> List[Image.Image]:
        embedded_tokens = self.tokenizer.embedding(self.obs_tokens)     # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]
