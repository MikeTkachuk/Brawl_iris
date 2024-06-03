from collections import OrderedDict
import cv2
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn as nn

from src.episode import Episode


def configure_optimizer(model, learning_rate, weight_decay, *blacklist_module_names):
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
                no_decay.add(fpn)
            elif 'bias' in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(
        param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def extract_state_dict(state_dict, module_name):
    return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def remove_dir(path, should_ask=False):
    assert path.is_dir()
    if (not should_ask) or input(f"Remove directory : {path} ? [Y/n] ").lower() != 'n':
        shutil.rmtree(path)


def compute_lambda_returns(rewards, values, ends, gamma, lambda_):
    assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
    assert rewards.shape == ends.shape == values.shape, f"{rewards.shape}, {values.shape}, {ends.shape}"  # (B, T, 1)
    t = rewards.size(1)
    lambda_returns = torch.empty_like(values)
    lambda_returns[:, -1] = values[:, -1]
    lambda_returns[:, :-1] = rewards[:, :-1] + ends[:, :-1].logical_not() * gamma * (1 - lambda_) * values[:, 1:]

    last = lambda_returns[:, -1]
    for i in list(range(t - 1))[::-1]:
        lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, i]

    return lambda_returns


@torch.no_grad()
def adaptive_gradient_clipping(parameters, lam=0.15):
    ratios = []
    for param in parameters:
        if param.grad is None or not param.requires_grad:
            continue
        ratio = torch.abs(param.grad) / (torch.abs(param) + 1E-3)
        ratios.append(ratio)
        param.grad = torch.where(ratio > lam, lam * param.grad / ratio, param.grad)
    return ratios


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self._kwargs = kwargs
        self._intermediate_losses = None

    @property
    def intermediate_losses(self):  # avoids gpu sync
        if self._intermediate_losses is None:
            self._intermediate_losses = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in
                                         self._kwargs.items()}

        return self._intermediate_losses

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self

    def __getitem__(self, item):
        return self._kwargs[item]


class EpisodeDirManager:
    def __init__(self, episode_dir: Path, max_num_episodes: int) -> None:
        self.episode_dir = episode_dir
        self.episode_dir.mkdir(parents=False, exist_ok=True)
        self.max_num_episodes = max_num_episodes
        self.best_return = float('-inf')

    def save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        if self.max_num_episodes is not None and self.max_num_episodes > 0:
            self._save(episode, episode_id, epoch)

    def _save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        ep_paths = [p for p in self.episode_dir.iterdir() if p.stem.startswith('episode_')]
        assert len(ep_paths) <= self.max_num_episodes
        if len(ep_paths) == self.max_num_episodes:
            to_remove = min(ep_paths, key=lambda ep_path: int(ep_path.stem.split('_')[1]))
            to_remove.unlink()
        episode.save(self.episode_dir / f'episode_{episode_id}_epoch_{epoch}.pt')

        ep_return = episode.compute_metrics().episode_return
        if ep_return > self.best_return:
            self.best_return = ep_return
            path_best_ep = [p for p in self.episode_dir.iterdir() if p.stem.startswith('best_')]
            assert len(path_best_ep) in (0, 1)
            if len(path_best_ep) == 1:
                path_best_ep[0].unlink()
            episode.save(self.episode_dir / f'best_episode_{episode_id}_epoch_{epoch}.pt')


class RandomHeuristic:
    def __init__(self, num_actions, num_continuous):
        self.num_actions = num_actions
        self.num_continuous = num_continuous

    def act(self, obs):
        assert obs.ndim == 4  # (N, H, W, C)
        n = obs.size(0)

        return torch.randint(low=0, high=self.num_actions, size=(n,)), torch.rand(size=(n, self.num_continuous))


def make_video(fname, fps, frames):
    assert frames.ndim == 4  # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3

    video = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        video.write(frame[:, :, ::-1])
    video.release()


class ActionTokenizer:
    def __init__(self, n_binary_actions=3, move_shot_anchors=(4, 4)):
        self.n_binary_actions = n_binary_actions
        self.move_shot_anchors = move_shot_anchors if hasattr(move_shot_anchors, "__len__") else (
                                                                                                 move_shot_anchors,) * 2

        self.n_actions = 2 ** self.n_binary_actions * (1 + move_shot_anchors[0]) * move_shot_anchors[1]
        self.bit_space = (len(bin(self.move_shot_anchors[0])) - 2,
                          len(bin(self.move_shot_anchors[1] - 1)) - 2)  # anchors only

        self._action_binary_maps = None

    def _action_map_binary(self, token, input_binary=False):
        """
        Converts between binary and consecutive formats. Binary is ready to be split into bins and parsed.
        Consecutive is basically an id of an action.
        :param token: int, action token
        :param input_binary: if True, assumes token to be in binary format
        :return: token in the opposite format
        """
        if self._action_binary_maps is None:
            b_tokens = []
            total_bits = self.n_binary_actions + sum(self.bit_space)
            for i in range(2 ** total_bits):
                bin_i = bin(i)[2:]
                bin_i = '0' * (total_bits - len(bin_i)) + bin_i
                bin_move_anchor = bin_i[self.n_binary_actions:][:self.bit_space[0]]
                if int(bin_move_anchor, 2) > self.move_shot_anchors[0]:
                    continue
                bin_shot_anchor = bin_i[-self.bit_space[1]:]
                if int(bin_shot_anchor, 2) >= self.move_shot_anchors[1]:
                    continue
                b_tokens.append(i)
            c_tokens = range(len(b_tokens))
            self._action_binary_maps = (
                dict(zip(c_tokens, b_tokens)),
                dict(zip(b_tokens, c_tokens))
            )

        if input_binary:
            return self._action_binary_maps[1][token]
        else:
            return self._action_binary_maps[0][token]

    def create_action_token(self, make_move, make_shot, super_ability, use_gadget,
                            move_anchor, shot_anchor, ):
        assert 0 <= move_anchor < self.move_shot_anchors[0]
        assert 0 <= shot_anchor < self.move_shot_anchors[1]
        move_anchor = move_anchor + 1 if make_move else 0

        bin_str = ''
        for i in [make_shot, super_ability, use_gadget]:
            bin_str += str(int(i))

        anch_bin = bin(move_anchor)[2:]
        bin_str += '0' * (self.bit_space[0] - len(anch_bin)) + anch_bin

        anch_bin = bin(shot_anchor)[2:]
        bin_str += '0' * (self.bit_space[1] - len(anch_bin)) + anch_bin

        assert len(bin_str) == self.n_binary_actions + sum(self.bit_space), "Incorrect token range"
        b_token = int(bin_str, 2)
        return self._action_map_binary(b_token, input_binary=True)

    def split_into_bins(self, token: int, input_binary=False, decode_move_anchor=True):
        if not input_binary:
            token = self._action_map_binary(token, input_binary=False)
        out = []
        bin_t = bin(token)[2:]
        total_bits = self.n_binary_actions + sum(self.bit_space)
        bin_t = '0' * (total_bits - len(bin_t)) + bin_t  # pad with 0

        for i in range(self.n_binary_actions):
            out.append(int(bin_t[i]))
        move_anchor = int(bin_t[self.n_binary_actions:][:self.bit_space[0]], 2)
        shot_anchor = int(bin_t[-self.bit_space[1]:], 2)
        out += [move_anchor, shot_anchor]
        if decode_move_anchor:  # add legacy make_move
            out = [int(move_anchor > 0)] + out
            if move_anchor > 0:
                out[-2] -= 1  # make move anchors in 0-n
        return out

    def parse_action_token(self, action):
        """
        Parse 1 consecutive action token and 3 continuous action values
        :param action: array-like of action values
        :return: dict of parsed actions
        """
        assert len(action) == 4, "Wrong action format"  # 1 token + 3 continuous [0,1]: (move, shot, strength)
        assert self.n_binary_actions == 3, "Only 3 binary actions supported. See docs"
        assert 0 <= action[0] < self.n_actions, "Action token out of bound"
        bin_action = self._action_map_binary(action[0])
        bins = bin(int(bin_action))[2:]
        total_bits = self.n_binary_actions + sum(self.bit_space)
        bins = '0' * (total_bits - len(bins)) + bins  # pad with 0

        make_shot, super_ability, use_gadget = bins[:self.n_binary_actions]

        move_anchor = int(bins[self.n_binary_actions:][:self.bit_space[0]], 2)
        shot_anchor = int(bins[-self.bit_space[1]:], 2)

        def _get_anchor_dir(anchor_num, total, shift=0.0):
            assert anchor_num < total
            angle_shift = 1 / total * 2 * np.pi * (shift - 0.5)  # assumes shift is in [0, 1]
            angle = anchor_num / total * 2 * np.pi + angle_shift
            anchor = np.array([np.cos(angle), np.sin(angle)])
            return anchor

        parsed_action = {
            'direction': _get_anchor_dir(max(move_anchor - 1, 0), self.move_shot_anchors[0], action[1]),  # 0 is no_move
            'make_move': int(move_anchor > 0),
            'make_shot': int(make_shot),
            'shoot_direction': _get_anchor_dir(shot_anchor, self.move_shot_anchors[1], action[2]),
            'shoot_strength': action[3],
            'super_ability': int(super_ability),
            'use_gadget': int(use_gadget),
        }
        return parsed_action


if __name__ == "__main__":
    a = ActionTokenizer()
    cont = [0.5, 0.5, 0.0]
    token = a.create_action_token(0,1,1,0,1,3)
    print(token)
    print(a.parse_action_token([token, *cont]))