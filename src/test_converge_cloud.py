import os
import shutil
import sys
from threading import Thread
from typing import Dict

sys.path.append(os.getcwd())

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import tqdm
import hydra
from omegaconf import DictConfig

from src.dataset import EpisodesDataset
from src.episode import Episode
from src.trainer import Trainer
from src.utils import compute_lambda_returns, LossWithIntermediateLosses
from src.models.actor_critic import ActorCriticOutput, ImagineOutput, ActorCritic

MAX_EPISODES = 4000
BATCH_SIZE = 4
WEIGHT_QUANTILE = 0.5  # value loss quantile to assign episode weights
RANDOM_ACTION = 0.05  # proba
POSITIVE_WEIGHT = 1.0  # weigh positive updates (divides neg ones)

# align weights so that positive episode with loss 0.1 was on par with the
# negative one via loss + neg_ep_w_shift
NEGATIVE_EPISODE_WEIGHT_SHIFT = 0.10

# set action_limit_mask to False and the desired actions to lock them during the run.

# make_move, make_shot, super_ability, use_gadget, move_anchor, shot_anchor
ACTION_LOCK_MASK = torch.tensor([True, False, False, False, True, False])
ACTION_LOCK = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# move_shift, shot_shift, shot_strength
ACTION_C_LOCK_MASK = torch.tensor([True, False, False])
ACTION_C_LOCK = torch.tensor([0.0, 0.0, -10.0])


def limit_speed(speed_constraint=2):
    def decorator(func):
        def inner(*args, **kwargs):
            start = time.time()
            out = func(*args, **kwargs)
            extra_time = (1 / speed_constraint) - (time.time() - start) if speed_constraint is not None else 0.0
            if extra_time > 0:
                time.sleep(extra_time)
            else:
                # print(f"off rate by {-extra_time}")
                pass
            return out

        return inner

    return decorator


def compute_masked_lambda_returns(rewards,
                                  values,
                                  ends,
                                  mask_paddings,
                                  gamma=0.995,
                                  lambda_=0.95,
                                  ):
    lambda_returns = torch.zeros_like(values)
    for b in range(rewards.size(0)):
        rewards_masked = rewards[b][mask_paddings[b]]
        values_masked = values[b][mask_paddings[b]]
        ends_masked = ends[b][mask_paddings[b]]
        if mask_paddings[b].count_nonzero():
            lambda_returns[b][mask_paddings[b]] = compute_lambda_returns(rewards_masked[None, ...],
                                                                         values_masked[None, ...],
                                                                         ends_masked[None, ...],
                                                                         gamma,
                                                                         lambda_)
    return lambda_returns


class ACTrainer:
    def __init__(self, env, actor: ActorCritic, optimizer, dataset: EpisodesDataset):
        self.env = env
        self.actor = actor
        self.optimizer = optimizer
        self.dataset = dataset

        self.actions, self.rewards, self.dones, self.outputs, self.mask_paddings = [], [], [], [], []
        self.observations = []
        self.metrics = defaultdict(float)

        self.batch_size = BATCH_SIZE

        self._step = 0
        self._pre_sample = None
        self._replay = None
        self._active_episodes = None
        self._update_weights = None

    def get_episode_proba(self, rank=True, alpha=0.7, beta=0.5):
        episode_weights = np.array([getattr(ep, 'weight', 0.0) for ep in self.dataset.episodes])
        if rank:
            criterion_rank = np.argsort(episode_weights, )
            criterion = np.zeros_like(episode_weights)
            criterion[criterion_rank] = 1 / (len(episode_weights) - np.arange(len(episode_weights)))
        else:
            criterion = episode_weights + 1E-3

        assert all(criterion > 0.0)

        probas = criterion ** alpha
        probas = probas / np.sum(probas)
        update_weights = 1 / (len(self.dataset) * probas) ** beta
        return probas, update_weights

    @torch.no_grad()
    def _assign_episode_weights(self, loss, mask_padding):
        for i, episode in enumerate(self._active_episodes):
            if episode is None or not mask_padding[i].any():
                continue
            episode_loss = loss[i][mask_padding[i]]  # for scale with what is logged
            old_weight = episode.weight
            new_weight = torch.quantile(episode_loss, WEIGHT_QUANTILE).detach().item()
            episode.weight = new_weight
            print(f"Changed weight: {old_weight:.3f} -> {new_weight:.3f} diff: {new_weight - old_weight:.3f}")

    def reset(self, ids_only=False, size=None, start_id=0):
        self.metrics = defaultdict(float)
        self._step = 0

        # sample episodes via prioritized sampling
        if ids_only or self._pre_sample is None:
            probas, weights = self.get_episode_proba()
            episode_ids = np.random.choice(np.arange(len(self.dataset)),
                                           size=(self.batch_size,),
                                           p=probas,
                                           replace=False)
            self._pre_sample = (probas, weights, episode_ids)
        if not ids_only:
            probas, weights, episode_ids = self._pre_sample
            size = size or self.batch_size
            self._active_episodes = [self.dataset.episodes[i] for i in episode_ids[start_id:start_id+size]]
            self._update_weights = weights[episode_ids[start_id:start_id+size]]
            self._replay = self._to_device(self.dataset.sample_replay(samples=episode_ids[start_id:start_id+size]))

        self.actions, self.rewards, self.dones, self.outputs, self.mask_paddings = [], [], [], [], []
        self.observations = []
        self.actor.reset(size or self.batch_size)
        self.actor.train()

    def _update_metrics(self, metrics: dict):
        for k, v in metrics.items():
            name = f"actor_critic/train/{k}"
            self.metrics[name] += v

    def episode_end(self, gradient_accumulation=True):
        for p in self.actor.parameters():
            p.grad = None

        splits = range(self.batch_size)
        if not gradient_accumulation:
            splits = [list(splits)]  # make splits[0] whole batch
        else:
            splits = [[s] for s in splits]
        for split in splits:
            self.reset(ids_only=False, size=len(split), start_id=split[0])
            for i in range(0, self._replay['ends'].size(1)):
                full_mask_padding = self._replay['mask_padding'][:len(split), i]
                self.mask_paddings.append(full_mask_padding)

                obs = self._replay['observations'][:len(split), i]
                output = self.actor.forward(obs, mask_padding=full_mask_padding)
                self.outputs.append(output)
                replay_actions = torch.cat([self._replay['actions'][:len(split), self._step],
                                            self._replay['actions_continuous'][:len(split), self._step]], dim=-1)
                full_actions = replay_actions
                self.actions.append(full_actions)
                full_rewards = self._replay['rewards'][:len(split), i]
                self.rewards.append(full_rewards)
                full_ends = self._replay['ends'][:len(split), i]
                self.dones.append(full_ends)

            episode_output = ImagineOutput(
                observations=None,
                actions=torch.stack(self.actions, dim=1)[..., :-3],
                actions_continuous=torch.stack(self.actions, dim=1)[..., -3:],
                logits_actions=torch.cat([out.logits_actions for out in self.outputs], dim=1),
                continuous_means=torch.cat([out.mean_continuous for out in self.outputs], dim=1),
                continuous_stds=torch.cat([out.std_continuous for out in self.outputs], dim=1),
                values=torch.stack([out.means_values for out in self.outputs], dim=1).reshape(len(split), -1),
                rewards=torch.stack(self.rewards, dim=1).reshape(len(split), -1),
                ends=torch.stack(self.dones, dim=1).reshape(len(split), -1)
            )

            loss = self.ac_loss(episode_output, torch.stack(self.mask_paddings, dim=1),) / self.batch_size
            loss.loss_total.backward()
            self._update_metrics(loss.intermediate_losses)
            self.metrics["actor_critic/train/total_loss"] += loss.loss_total.item()
        self.optimizer.step()
        return self.metrics

    def ac_loss(self, outputs: ImagineOutput, mask_paddings, entropy_weight=0.001):
        with torch.no_grad():
            lambda_returns = compute_masked_lambda_returns(
                rewards=outputs.rewards,
                values=outputs.values,
                ends=outputs.ends,
                mask_paddings=mask_paddings,
                gamma=0.995,
                lambda_=0.95,
            )
        mask_paddings = torch.logical_and(mask_paddings, outputs.ends.logical_not())  # do not include end into loss
        values = outputs.values

        (log_probs, entropy), (log_probs_continuous, entropy_cont) = self.actor.get_proba_entropy(outputs)

        batch_lengths = torch.count_nonzero(mask_paddings, dim=1).unsqueeze(-1)  # TODO questionable
        update_weight = torch.tensor(self._update_weights, device=self.device).unsqueeze(1) * \
                        torch.min(batch_lengths[batch_lengths > 0]) / batch_lengths
        advantage_factor = lambda_returns - values.detach()
        advantage_factor[advantage_factor < 0] /= POSITIVE_WEIGHT

        # compute losses
        loss_actions = -1 * (log_probs * advantage_factor.unsqueeze(-1))[..., ACTION_LOCK_MASK]
        loss_actions_masked = torch.masked_select(update_weight.unsqueeze(-1) * loss_actions,
                                                  mask_paddings.unsqueeze(-1)).mean()

        loss_continuous_actions = -1 * (log_probs_continuous.clamp(-5, 5) * advantage_factor.unsqueeze(-1))[..., ACTION_C_LOCK_MASK]
        loss_continuous_actions_masked = torch.masked_select(update_weight.unsqueeze(-1) * loss_continuous_actions,
                                                             mask_paddings.unsqueeze(-1)).mean()

        loss_entropy = torch.masked_select(- entropy_weight * entropy, mask_paddings.unsqueeze(-1)).mean()
        loss_entropy_continuous = torch.masked_select(- entropy_weight * entropy_cont,
                                                      mask_paddings.unsqueeze(-1)).mean()
        loss_values = torch.square(values - lambda_returns)
        loss_values_masked = torch.masked_select(update_weight * loss_values, mask_paddings).mean()

        full_loss = LossWithIntermediateLosses(loss_actions=loss_actions_masked,
                                               loss_continuous_actions=loss_continuous_actions_masked,
                                               loss_values=loss_values_masked,
                                               loss_entropy=loss_entropy,
                                               loss_entropy_continuous=loss_entropy_continuous)
        # episode weight ~ absolute loss per episode
        loss_actions = loss_actions.detach()
        loss_actions[loss_actions < 0] = loss_actions[loss_actions < 0] + NEGATIVE_EPISODE_WEIGHT_SHIFT
        loss_continuous_actions = loss_entropy_continuous.detach()
        loss_continuous_actions[loss_continuous_actions < 0] = loss_continuous_actions[
                                                                   loss_continuous_actions < 0
                                                                   ] + NEGATIVE_EPISODE_WEIGHT_SHIFT

        self._assign_episode_weights(loss_values + loss_actions.mean(dim=-1) + loss_continuous_actions.mean(dim=-1),
                                     mask_paddings)
        return full_loss

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}

    @property
    def device(self):
        return self.actor.device


@hydra.main(config_path=r"../config", config_name="trainer")
def main_replay(cfg: DictConfig):
    os.system(
        f"aws s3 cp s3://brawl-stars-iris/pretrain_data/test_converge/ {os.getcwd()}/checkpoints "
        f"--exclude \"*\" --include \"dataset/*\" --recursive --quiet")
    shutil.copytree(r'/home/ec2-user/checkpoints', os.getcwd() + r'/checkpoints', dirs_exist_ok=True)

    trainer = Trainer(cfg, cloud_instance=True)
    trainer.run_prefix = cfg.run_prefix
    trainer.load_checkpoint()

    # import shutil
    # shutil.copytree(r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\outputs\2023-08-18\22-50-03\checkpoints",
    #                 "checkpoints", dirs_exist_ok=True)
    # trainer = Trainer(cfg, cloud_instance=True)
    # trainer.save_checkpoint(0, save_agent_only=False, save_dataset=False)
    # trainer.load_checkpoint()

    env = None
    actor = trainer.agent.actor_critic
    optimizer = trainer.optimizer_actor_critic
    ac_trainer = ACTrainer(env, actor, optimizer, trainer.train_dataset)
    ac_trainer.batch_size = cfg.training.actor_critic.batch_num_samples

    metrics = []
    for n_step in tqdm.tqdm(range(MAX_EPISODES // ac_trainer.batch_size), desc="Step: ",
                            total=MAX_EPISODES // ac_trainer.batch_size, file=sys.stdout):
        ac_trainer.reset(ids_only=True)
        episode_metrics = ac_trainer.episode_end(gradient_accumulation=True)
        metrics += [{"epoch": n_step, **episode_metrics}]
        if n_step % round(100 / ac_trainer.batch_size) == 0:
            print('Saving checkpoint', flush=True)
            # save metrics
            metrics_file_path = Path(trainer.cfg.cloud.log_metrics).name
            with open(metrics_file_path, 'w') as metrics_file:
                json.dump(metrics, metrics_file)
                metrics = []

            os.system(
                f"aws s3 cp {metrics_file_path} s3://{trainer.cfg.cloud.bucket_name}/{trainer.cfg.cloud.log_metrics}")
            trainer.save_checkpoint(n_step, save_agent_only=False, save_dataset=False)
            os.system(f'aws s3 cp checkpoints s3://brawl-stars-iris/{trainer.run_prefix}/checkpoints '
                      f'--recursive '
                      f'--exclude "dataset/*"')


if __name__ == "__main__":
    main_replay()
