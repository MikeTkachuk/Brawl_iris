from dataclasses import dataclass
from typing import Any, Optional, Union
import sys
from functools import partial

from einops import rearrange
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm import tqdm

from src.dataset import Batch
from src.envs.world_model_env import WorldModelEnv
from src.models.tokenizer import Tokenizer
from src.models.tokenizer.nets import Normalize
from src.models.world_model import WorldModel
from src.utils import compute_lambda_returns, LossWithIntermediateLosses, ActionTokenizer


@dataclass
class ActorCriticOutput:
    logits_actions: torch.FloatTensor
    mean_continuous: torch.FloatTensor
    std_continuous: torch.FloatTensor
    means_values: torch.FloatTensor


@dataclass
class ImagineOutput:
    observations: torch.ByteTensor
    actions: torch.LongTensor
    actions_continuous: torch.LongTensor
    logits_actions: torch.FloatTensor
    continuous_means: torch.FloatTensor
    continuous_stds: torch.FloatTensor
    values: torch.FloatTensor
    rewards: torch.FloatTensor
    ends: torch.BoolTensor


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int = None, conv_shortcut: bool = False,
                 dropout: float, nonlinearity=nn.Mish()) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=False)
        self.norm1 = Normalize(out_channels)

        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=False)
        self.norm2 = Normalize(out_channels)

        self.nonlinearity = nonlinearity
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.conv1(h)
        h = self.norm1(h)
        h = self.nonlinearity(h)

        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return self.nonlinearity(x + h)


class Backbone(nn.Module):
    def __init__(self, channels=None, kernels=None, strides=None, paddings=None, pools=None, resnet_blocks=None,
                 activation=nn.Mish()):
        super().__init__()
        channels = channels or [3, 64, 80, 128, 256, 256, 256]  # [3, 32, 32, 48, 48, 64, 64, 64]
        kernels = kernels or [7, 5, 5, 3, 3, 3]
        strides = strides or [1, 1, 1, 1, 1, 1]
        paddings = paddings or [3, 2, 2, 1, 1, 1]
        pools = pools or [2, 2, 2, 2, 2, 2]
        resnet_block = resnet_blocks or [False, False, True, True, True, True]

        self.out_channels = channels[-1]
        self.out_width = 3  # for 192 input size
        self.blocks = nn.ModuleList()

        for i, (st, ker, pad, pool, is_res_block) in enumerate(zip(strides, kernels, paddings, pools, resnet_block)):
            block = nn.Sequential()
            if is_res_block:
                block.append(ResnetBlock(in_channels=channels[i],
                                         out_channels=channels[i + 1],
                                         dropout=0.01, nonlinearity=activation))
            else:
                conv = nn.Conv2d(channels[i],
                                 channels[i + 1],
                                 kernel_size=ker,
                                 stride=st,
                                 padding=pad,
                                 bias=False)
                block.append(conv)
                block.append(Normalize(channels[i + 1]))
                block.append(activation)
            if pool == 1:
                block.append(nn.Identity())
            else:
                block.append(nn.MaxPool2d(pool, pool))

            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


class ActorCritic(nn.Module):
    def __init__(self, move_shot_anchors,
                 act_continuous_size,
                 act_lock=None,
                 act_continuous_lock=None,
                 use_original_obs: bool = False,
                 checkpoint_backbone=False,
                 checkpoint_lstm=False,
                 fp16=False,
                 activation=nn.Mish()) -> None:
        super().__init__()
        self.use_original_obs = use_original_obs
        self.checkpoint_backbone = checkpoint_backbone
        self.checkpoint_lstm = checkpoint_lstm
        self.fp16 = fp16
        self.activation = activation

        self.backbone = Backbone(activation=self.activation)
        self.backbone_norm = Normalize(in_channels=self.backbone.out_channels)

        self.emb_dim = self.backbone.out_channels * self.backbone.out_width ** 2
        self.lstm_dim = 512
        self.lstm = nn.LSTMCell(self.emb_dim, self.lstm_dim)
        self.hx, self.cx = None, None

        self.emb_norm = Normalize(in_channels=self.lstm_dim)
        self.emb_linear = nn.Linear(self.emb_dim + self.lstm_dim, self.lstm_dim)
        self.act_tokenizer = ActionTokenizer(move_shot_anchors=move_shot_anchors)
        self.act_vocab_size = self.act_tokenizer.n_actions
        self.act_continuous_size = act_continuous_size
        self.act_lock = act_lock
        self.act_continuous_lock = act_continuous_lock
        self.actor_head = nn.Sequential(
            nn.Linear(self.lstm_dim, 128),
            Normalize(128), self.activation,

            nn.Linear(128, 64),
            Normalize(64), self.activation,

            nn.Linear(64, self.act_vocab_size + 2 * self.act_continuous_size, bias=False
                      )
        )
        self.critic_head = nn.Sequential(
            nn.Linear(self.lstm_dim, 1, bias=False),
        )

    def __repr__(self) -> str:
        return "actor_critic"

    @property
    def device(self):
        return self.backbone.parameters().__next__().device

    def clear(self) -> None:
        self.hx, self.cx = None, None

    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None,
              mask_padding: Optional[torch.Tensor] = None) -> None:
        device = self.backbone.blocks[0][0].weight.device
        dtype = torch.float16 if self.fp16 else torch.float32
        self.hx = torch.zeros(n, self.lstm_dim, dtype=dtype, device=device)
        self.cx = torch.zeros(n, self.lstm_dim, dtype=dtype, device=device)
        if burnin_observations is not None:
            assert burnin_observations.ndim == 5 and burnin_observations.size(
                0) == n and mask_padding is not None and burnin_observations.shape[:2] == mask_padding.shape
            for i in range(burnin_observations.size(1)):
                if mask_padding[:, i].any():
                    with torch.no_grad():
                        self(burnin_observations[:, i], mask_padding[:, i])

    def prune(self, mask: np.ndarray) -> None:
        self.hx = self.hx[mask]
        self.cx = self.cx[mask]

    def forward(self, inputs: torch.FloatTensor, mask_padding: Optional[torch.BoolTensor] = None) -> ActorCriticOutput:
        with torch.amp.autocast(self.device.type, enabled=self.fp16):
            assert inputs.ndim == 4  # and inputs.shape[1:] == (3, 64, 64)
            assert 0 <= inputs.min() <= 1 and 0 <= inputs.max() <= 1
            assert mask_padding is None or (
                    mask_padding.ndim == 1 and mask_padding.size(0) == inputs.size(0))

            if mask_padding is not None and not mask_padding.any():
                # noinspection PyTypeChecker
                return ActorCriticOutput(
                    logits_actions=torch.zeros((mask_padding.size(0), 1, self.act_vocab_size),
                                               dtype=self.hx.dtype, device=self.device),
                    mean_continuous=torch.zeros((mask_padding.size(0), 1, self.act_continuous_size),
                                                dtype=self.hx.dtype, device=self.device),
                    std_continuous=torch.ones((mask_padding.size(0), 1, self.act_continuous_size),
                                              dtype=self.hx.dtype, device=self.device),
                    means_values=torch.zeros((mask_padding.size(0), 1, 1),
                                             dtype=self.hx.dtype, device=self.device)
                )

            x = inputs[mask_padding] if mask_padding is not None else inputs

            x = x.mul(2).sub(1)
            if self.checkpoint_backbone:
                x = torch.utils.checkpoint.checkpoint(self.backbone, x, use_reentrant=False)
            else:
                x = self.backbone(x)

            x = self.backbone_norm(x)
            x = torch.flatten(x, start_dim=1)
            if self.checkpoint_lstm:
                lstm_func = partial(torch.utils.checkpoint.checkpoint, self.lstm, use_reentrant=False)
            else:
                lstm_func = self.lstm
            if mask_padding is None:
                self.hx, self.cx = lstm_func(x, (self.hx, self.cx))
                hx_normed = self.emb_norm(self.hx)
                x = self.emb_linear(torch.cat([x, hx_normed], dim=-1))
            else:
                self.hx[mask_padding], self.cx[mask_padding] = lstm_func(x,
                                                                         (self.hx[mask_padding], self.cx[mask_padding]))
                hx_normed = self.emb_norm(self.hx[mask_padding])
                x = self.emb_linear(torch.cat([x, hx_normed], dim=-1))

            x = self.activation(x)

            full_logits = rearrange(self.actor_head(x), 'b a -> b 1 a')
            means_values = rearrange(self.critic_head(x), 'b 1 -> b 1 1')

            if mask_padding is not None:
                full_logits_placeholder = torch.zeros((self.hx.size(0), *full_logits.shape[1:]),
                                                      dtype=full_logits.dtype,
                                                      device=full_logits.device)
                full_logits_placeholder[mask_padding] = full_logits
                full_logits = full_logits_placeholder

                means_values_placeholder = torch.zeros((self.hx.size(0), *means_values.shape[1:]),
                                                       dtype=means_values.dtype,
                                                       device=means_values.device)
                means_values_placeholder[mask_padding] = means_values
                means_values = means_values_placeholder

            logits_actions = full_logits[...,
                             :self.act_vocab_size]
            mean_continuous = full_logits[..., self.act_vocab_size:self.act_vocab_size + self.act_continuous_size]
            mean_continuous = 5 * F.tanh(mean_continuous / 5)  # soft clamp logits to [-5, 5]
            std_continuous = full_logits[..., self.act_vocab_size + self.act_continuous_size:]
            # soft clamp to [1E-1, 5]
            scale = (5 - 1E-1) / 2
            shift = (5 + 1E-1) / 2
            std_continuous = scale * F.tanh(std_continuous / scale - 0.7) + shift  # f(-0.7) = 1.07

            return ActorCriticOutput(logits_actions, mean_continuous, std_continuous, means_values)

    def sample_actions(self, out: ActorCriticOutput, eps=0.01):
        d = Categorical(logits=out.logits_actions.unsqueeze(1))
        action = d.sample()
        if torch.rand(1) < eps:
            action = torch.randint(0, out.logits_actions.size(-1),
                                   size=action.shape, device=out.logits_actions.device)

        d_cont = Normal(out.mean_continuous, out.std_continuous)
        action_cont = d_cont.rsample()

        def _lock_one(act_token: int):
            factorized = self.act_tokenizer.split_into_bins(act_token)
            assert len(factorized) == len(self.act_lock)
            for i in range(len(factorized)):
                if self.act_lock[i] is not None:
                    factorized[i] = self.act_lock[i]
            return self.act_tokenizer.create_action_token(*factorized)

        if self.act_lock is not None:
            action = action.to("cpu").apply_(_lock_one).to(self.device)
        if self.act_continuous_lock is not None:
            mask = torch.tensor([i is not None for i in self.act_continuous_lock], device=self.device)
            action_cont = torch.where(mask, torch.tensor(self.act_continuous_lock, device=self.device), action_cont)
        return action, action_cont

    @staticmethod
    def get_proba_entropy(out: ImagineOutput):
        d = Categorical(logits=out.logits_actions[:, :-1])
        prob = d.log_prob(out.actions[:, :-1])
        ent = d.entropy()
        d_cont = Normal(out.continuous_means[:, :-1], out.continuous_stds[:, :-1])
        return (prob, ent), (d_cont.log_prob(out.actions_continuous[:, :-1]), d_cont.entropy())

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, imagine_horizon: int,
                     gamma: float, lambda_: float, entropy_weight: float, should_imagine=True, **kwargs: Any):
        if not self.use_original_obs and should_imagine:
            outputs = self.imagine(batch, tokenizer, world_model, horizon=imagine_horizon)
        else:
            outputs = self.recall(batch, imagine_horizon)

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=outputs.values,
                ends=outputs.ends,
                gamma=gamma,
                lambda_=lambda_,
            )[:, :-1]

        values = outputs.values[:, :-1]
        advantages = lambda_returns - values.detach()
        (log_probs, entropy), (log_probs_continuous, entropy_continuous) = self.get_proba_entropy(outputs)
        loss_actions = -1 * (
                log_probs * advantages).mean()

        loss_continuous_actions = -1 * (
                log_probs_continuous * advantages.unsqueeze(-1)).mean()

        loss_entropy = - entropy_weight * entropy.mean()
        loss_entropy_continuous = - entropy_weight * entropy_continuous.mean()
        loss_values = F.mse_loss(values, lambda_returns)

        return LossWithIntermediateLosses(
                                          # losses
                                          loss_actions=loss_actions,
                                          # loss_continuous_actions=loss_continuous_actions,
                                          loss_values=loss_values,
                                          loss_entropy=loss_entropy,
                                          # loss_entropy_continuous=loss_entropy_continuous,

                                          # metrics
                                          values_mean=values.mean().detach(),
                                          advantage_mean=advantages.mean().detach(),
                                          custom_actions=torch.where(
                                            advantages >= 0,
                                            -log_probs,
                                            -torch.log(1 - torch.exp(log_probs))).mean().detach(),
                                          custom_actions_weighted=torch.where(
                                            advantages >= 0,
                                            -log_probs*advantages,
                                            advantages*torch.log(1 - torch.exp(log_probs))).mean().detach()
                                          ), outputs

    def imagine(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, horizon: int,
                show_pbar: bool = False) -> ImagineOutput:
        assert not self.use_original_obs
        initial_observations = batch['observations']
        mask_padding = batch['mask_padding']
        assert initial_observations.ndim == 5  # and initial_observations.shape[2:] == (3, 64, 64)
        assert mask_padding[:, -1].all()
        device = initial_observations.device
        wm_env = WorldModelEnv(tokenizer, world_model, device)

        all_actions = []
        all_continuous = []
        all_logits_actions = []
        all_continuous_means = []
        all_continuous_stds = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []

        burnin_observations = torch.clamp(
            tokenizer.encode_decode(initial_observations[:, :-1], should_preprocess=True, should_postprocess=True), 0,
            1) if initial_observations.size(1) > 1 else None
        self.reset(n=initial_observations.size(0), burnin_observations=burnin_observations,
                   mask_padding=mask_padding[:, :-1])  # TODO wm receives much less context than the actor

        obs = wm_env.reset_from_initial_observations(initial_observations[:, -1])
        for k in tqdm(range(horizon), disable=not show_pbar, desc='Imagination', file=sys.stdout):
            all_observations.append(obs)

            outputs_ac = self(obs)
            action_token = Categorical(
                logits=outputs_ac.logits_actions).sample()

            action_continuous = Normal(outputs_ac.mean_continuous, outputs_ac.std_continuous).rsample()
            obs, reward, done, _ = wm_env.step(action_token,
                                               continuous=action_continuous)
            reward = torch.tensor(reward).reshape(-1, 1)
            done = torch.tensor(done).reshape(-1, 1)
            # lambda_return zeroes last step rewards
            if all_rewards:
                all_rewards[-1][done] += reward[done]
            all_actions.append(action_token)
            all_continuous.append(action_continuous)
            all_logits_actions.append(outputs_ac.logits_actions)
            all_continuous_means.append(outputs_ac.mean_continuous)
            all_continuous_stds.append(outputs_ac.std_continuous)
            all_values.append(outputs_ac.means_values)
            all_rewards.append(reward)
            all_ends.append(done)

        self.clear()

        return ImagineOutput(
            observations=torch.stack(all_observations, dim=1).mul(255).byte(),  # (B, T, C, H, W) in [0, 255]
            actions=torch.cat(all_actions, dim=1),  # (B, T)
            actions_continuous=torch.cat(all_continuous, dim=1),  # (B, T, #actions)
            logits_actions=torch.cat(all_logits_actions, dim=1),  # (B, T, #actions)
            continuous_means=torch.cat(all_continuous_means, dim=1),  # (B, T, #actions)
            continuous_stds=torch.cat(all_continuous_stds, dim=1),  # (B, T, #actions)
            values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),  # (B, T)
            rewards=torch.cat(all_rewards, dim=1).to(device),  # (B, T)
            ends=torch.cumsum(torch.cat(all_ends, dim=1), dim=-1).bool().to(device),  # (B, T)  propagate end
        )

    def recall(self, batch: Batch, horizon: int, show_pbar: bool = False):
        # burn-in param workaround: new_burn_in = burn_in - horizon
        total_n = batch['observations'].size(1)
        initial_observations = batch['observations'][:, :total_n - horizon - 1]
        init_mask_padding = batch['mask_padding'][:, :total_n - horizon - 1]
        assert initial_observations.ndim == 5
        assert batch['mask_padding'][:, -1].all()
        device = initial_observations.device
        self.reset(n=initial_observations.size(0), burnin_observations=initial_observations,
                   mask_padding=init_mask_padding)

        all_actions = []
        all_continuous = []
        all_logits_actions = []
        all_continuous_means = []
        all_continuous_stds = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []

        for k in tqdm(range(total_n - horizon - 1, total_n), disable=not show_pbar, desc='Recalling', file=sys.stdout):
            obs = batch['observations'][:, k]
            all_observations.append(obs)

            outputs_ac = self(obs)

            reward = batch['rewards'][:, k]
            done = batch['ends'][:, k]
            action = batch['actions'][:, k]
            continuous = batch['actions_continuous'][:, k]

            all_actions.append(action.reshape(-1, 1))
            all_continuous.append(continuous.unsqueeze(1))
            all_logits_actions.append(outputs_ac.logits_actions)
            all_continuous_means.append(outputs_ac.mean_continuous)
            all_continuous_stds.append(outputs_ac.std_continuous)
            all_values.append(outputs_ac.means_values)
            all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))

        self.clear()

        return ImagineOutput(
            observations=torch.stack(all_observations, dim=1).mul(255).byte(),  # (B, T, C, H, W) in [0, 255]
            actions=torch.cat(all_actions, dim=1),  # (B, T)
            actions_continuous=torch.cat(all_continuous, dim=1),  # (B, T, #actions)
            logits_actions=torch.cat(all_logits_actions, dim=1),  # (B, T, #actions)
            continuous_means=torch.cat(all_continuous_means, dim=1),  # (B, T, #actions)
            continuous_stds=torch.cat(all_continuous_stds, dim=1),  # (B, T, #actions)
            values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),  # (B, T)
            rewards=torch.cat(all_rewards, dim=1).to(device),  # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),  # (B, T)
        )


class SeparateActorCritic(ActorCritic):
    def __init__(self, critic: ActorCritic, critic_grad=False, **kwargs):
        super().__init__(0, 3, **kwargs)
        self.critic = critic
        self.critic_grad = critic_grad

    def clear(self) -> None:
        super().clear()
        self.critic.clear()

    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None,
              mask_padding: Optional[torch.Tensor] = None) -> None:
        super().reset(n, burnin_observations, mask_padding)
        if self.critic_grad:
            self.critic.reset(n, burnin_observations, mask_padding)
        else:
            with torch.no_grad():
                self.critic.reset(n, burnin_observations, mask_padding)

    def forward(self, inputs: torch.FloatTensor, mask_padding: Optional[torch.BoolTensor] = None):
        output = super().forward(inputs, mask_padding)
        if self.critic_grad:
            values = self.critic.forward(inputs, mask_padding)
        else:
            with torch.no_grad():
                values = self.critic.forward(inputs, mask_padding)
        output.means_values = values.means_values
        return output


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, normalize=False):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.normalize = normalize
        self.norm = Normalize(4 * self.hidden_dim)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        if self.normalize:
            combined_conv = self.norm(combined_conv)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvActorCritic(ActorCritic):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone = Backbone(channels=[3, 32, 64, 64, 128, 128, 128],
                                 pools=[2, 2, 2, 2, 2, 1], activation=self.activation)  # output.shape = (B, 128, 6, 6)
        self.backbone_norm = Normalize(in_channels=self.backbone.out_channels)

        self.emb_dim = self.backbone.out_channels
        self.lstm_dim = 64
        self.lstm = ConvLSTMCell(self.emb_dim, self.lstm_dim, (3, 3), normalize=True)
        self.hx, self.cx = None, None
        self.emb_norm = Normalize(self.lstm_dim)

        self.actor_head = nn.Sequential(
            nn.Conv2d(self.emb_dim + self.lstm_dim, self.lstm_dim, 3),
            Normalize(self.lstm_dim), self.activation,

            nn.Conv2d(self.lstm_dim, self.lstm_dim, 3),
            Normalize(self.lstm_dim), self.activation,
            nn.Flatten(),
            nn.LazyLinear(self.act_vocab_size + 2 * self.act_continuous_size, bias=False)
        )
        self.critic_head = nn.Sequential(
            nn.Conv2d(self.emb_dim + self.lstm_dim, self.lstm_dim, 3),
            Normalize(self.lstm_dim), self.activation,
            nn.Flatten(),
            nn.LazyLinear(1, bias=False),
        )

    def __repr__(self) -> str:
        return "convlstm_actor_critic"

    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None,
              mask_padding: Optional[torch.Tensor] = None) -> None:
        device = self.backbone.blocks[0][0].weight.device
        dtype = torch.float16 if self.fp16 else torch.float32
        self.hx = torch.zeros(n, self.lstm_dim, 8, 8, dtype=dtype, device=device)
        self.cx = torch.zeros(n, self.lstm_dim, 8, 8, dtype=dtype, device=device)
        if burnin_observations is not None:
            assert burnin_observations.ndim == 5 and burnin_observations.size(
                0) == n and mask_padding is not None and burnin_observations.shape[:2] == mask_padding.shape
            for i in range(burnin_observations.size(1)):
                if mask_padding[:, i].any():
                    with torch.no_grad():
                        self(burnin_observations[:, i], mask_padding[:, i])

    def forward(self, inputs: torch.FloatTensor, mask_padding: Optional[torch.BoolTensor] = None) -> ActorCriticOutput:
        with torch.amp.autocast(self.device.type, enabled=self.fp16):
            assert inputs.ndim == 4  # and inputs.shape[1:] == (3, 64, 64)
            assert 0 <= inputs.min() <= 1 and 0 <= inputs.max() <= 1
            assert mask_padding is None or (
                    mask_padding.ndim == 1 and mask_padding.size(0) == inputs.size(0))

            if mask_padding is not None and not mask_padding.any():
                # noinspection PyTypeChecker
                return ActorCriticOutput(
                    logits_actions=torch.zeros((mask_padding.size(0), 1, self.act_vocab_size),
                                               dtype=self.hx.dtype, device=self.device),
                    mean_continuous=torch.zeros((mask_padding.size(0), 1, self.act_continuous_size),
                                                dtype=self.hx.dtype, device=self.device),
                    std_continuous=torch.ones((mask_padding.size(0), 1, self.act_continuous_size),
                                              dtype=self.hx.dtype, device=self.device),
                    means_values=torch.zeros((mask_padding.size(0), 1, 1),
                                             dtype=self.hx.dtype, device=self.device)
                )

            x = inputs[mask_padding] if mask_padding is not None else inputs

            x = x.mul(2).sub(1)
            if self.checkpoint_backbone:
                x = torch.utils.checkpoint.checkpoint(self.backbone, x, use_reentrant=False)
            else:
                x = self.backbone(x)

            x = self.backbone_norm(x)
            if self.checkpoint_lstm:
                lstm_func = partial(torch.utils.checkpoint.checkpoint, self.lstm, use_reentrant=False)
            else:
                lstm_func = self.lstm
            if mask_padding is None:
                self.hx, self.cx = lstm_func(x, (self.hx, self.cx))
                hx_normed = self.emb_norm(self.hx)
            else:
                self.hx[mask_padding], self.cx[mask_padding] = lstm_func(x,
                                                                         (self.hx[mask_padding], self.cx[mask_padding]))
                hx_normed = self.emb_norm(self.hx[mask_padding])

            x = torch.cat([x, hx_normed], dim=1)

            full_logits = rearrange(self.actor_head(x), 'b a -> b 1 a')
            means_values = rearrange(self.critic_head(x), 'b 1 -> b 1 1')

            if mask_padding is not None:
                full_logits_placeholder = torch.zeros((self.hx.size(0), *full_logits.shape[1:]),
                                                      dtype=full_logits.dtype,
                                                      device=full_logits.device)
                full_logits_placeholder[mask_padding] = full_logits
                full_logits = full_logits_placeholder

                means_values_placeholder = torch.zeros((self.hx.size(0), *means_values.shape[1:]),
                                                       dtype=means_values.dtype,
                                                       device=means_values.device)
                means_values_placeholder[mask_padding] = means_values
                means_values = means_values_placeholder

            logits_actions = full_logits[...,
                             :self.act_vocab_size]
            mean_continuous = full_logits[..., self.act_vocab_size:self.act_vocab_size + self.act_continuous_size]
            mean_continuous = 5 * F.tanh(mean_continuous / 5)  # soft clamp logits to [-5, 5]
            std_continuous = full_logits[..., self.act_vocab_size + self.act_continuous_size:]
            # soft clamp to [1E-1, 5]
            scale = (5 - 1E-1) / 2
            shift = (5 + 1E-1) / 2
            std_continuous = scale * F.tanh(std_continuous / scale - 0.7) + shift  # f(-0.7) = 1.07

            return ActorCriticOutput(logits_actions, mean_continuous, std_continuous, means_values)


if __name__ == "__main__":
    import time


    def print_time(func, *args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        print(time.time() - start)


    actor = ActorCritic(1024, 3)
    actor.reset(1)
