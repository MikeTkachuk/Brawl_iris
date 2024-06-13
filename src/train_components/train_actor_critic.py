import math
import random
import shutil
import sys
from collections import deque
from pathlib import Path
import time
import os

SOURCE_ROOT = str(Path(__file__).absolute().parents[2])
sys.path.append(SOURCE_ROOT)

import hydra
import wandb
from hydra.utils import instantiate

import numpy as np
import torch
import torch.backends
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.dataset import EpisodesDataset
from src.utils import set_seed, adaptive_gradient_clipping, ActionTokenizer
from src.models.tokenizer.tokenizer import Tokenizer
from src.models.world_model import WorldModel
from src.models.actor_critic import ConvActorCritic, ImagineOutput
from src.episode import Episode

device = "cuda"


class SegmentDataset:
    def __init__(self, batch_size, save_dir=Path("./checkpoints/imagined"), max_len=15000):
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.max_len = max_len
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)

        self._samples = []

        self.load()

    def load(self):
        self._samples = sorted([int(p.stem) for p in self.save_dir.glob("*")])

    def __len__(self):
        return len(self._samples)

    @property
    def can_sample(self):
        return len(self) >= self.batch_size * 5

    def update(self, segments: ImagineOutput, burn_in: dict):
        size = segments.observations.size(0)
        for i in range(size):
            last_id = self._samples[-1] if len(self._samples) else -1
            episode = Episode(
                observations=torch.cat([burn_in["observations"][i, :-1].mul(255).clamp(0,255).byte(), segments.observations[i]], dim=0),
                actions=torch.cat([burn_in["actions"][i, :-1], segments.actions[i]], dim=0),
                actions_continuous=torch.cat([burn_in["actions_continuous"][i, :-1], segments.actions_continuous[i]], dim=0),
                ends=torch.cat([burn_in["ends"][i, :-1], segments.ends[i]], dim=0),
                rewards=torch.cat([burn_in["rewards"][i, :-1], segments.rewards[i]], dim=0),
                mask_padding=torch.cat([burn_in["mask_padding"][i, :-1], torch.ones_like(segments.ends[i], dtype=torch.bool)], dim=0),
                should_clip_end=False  # otherwise alignment of ends and burn-in start/end  will be a pain
            )
            episode.save(self.save_dir / f"{last_id + 1}.pt")
            self._samples.append(last_id + 1)

        num_overflow = max(0, len(self._samples) - self.max_len)
        if num_overflow:
            for i in range(num_overflow):
                (self.save_dir / f"{self._samples[i]}.pt").unlink()
            self._samples = self._samples[num_overflow:]

    def sample(self):
        alpha = 1.2
        assert self.can_sample
        l = self._samples[-1]
        ages = np.array([(l - s) % self.batch_size for s in self._samples])
        p = 1 / (1 + ages) ** alpha
        p /= p.sum()

        episode_ids = np.random.choice(self._samples,
                                       size=(self.batch_size,),
                                       p=p,
                                       replace=False)
        episodes = [Episode(**torch.load(self.save_dir / f"{e_id}.pt", map_location="cpu")) for e_id in episode_ids]
        max_len = max([len(ep) for ep in episodes])
        episodes = [ep.segment(len(ep)-max_len, len(ep), should_pad=True) for ep in episodes]
        return EpisodesDataset.collate_episodes_segments(episodes)


def read_only_dataset(dataset: EpisodesDataset) -> EpisodesDataset:
    dataset.add_episode = None
    dataset.update_disk_checkpoint = None
    return dataset


def get_dataset_split(cfg):
    dataset: EpisodesDataset = read_only_dataset(instantiate(cfg.datasets.train))
    dataset.load_disk_checkpoint(Path(SOURCE_ROOT) / "input_artifacts/dataset")
    episode_ids = dataset.disk_episodes
    split = train_test_split(episode_ids, test_size=0.1)
    train_dataset: EpisodesDataset = read_only_dataset(instantiate(cfg.datasets.train))
    train_dataset.disk_episodes = deque(split[0])
    eval_dataset: EpisodesDataset = read_only_dataset(instantiate(cfg.datasets.train))
    eval_dataset.disk_episodes = deque(split[1])

    train_dataset._dir = eval_dataset._dir = dataset._dir
    return train_dataset, eval_dataset


def custom_setup(cfg):
    torch.backends.cudnn.benchmark = True

    cfg.wandb.tags = list(set(cfg.wandb.tags or [] + ["actor_critic"]))
    wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
               reinit=True,
               resume=True,
               **cfg.wandb)


@hydra.main(config_path="../../config", config_name="actor_critic")
def main(cfg):
    try:
        print(f"PID: {os.getpid()}")
        set_seed(cfg.common.seed)
        if sys.gettrace() is not None:  # if debugging
            cfg.wandb.mode = "offline"
            cfg.training.actor_critic.batch_num_samples = 2

        Path("checkpoints/dataset").mkdir(parents=True, exist_ok=True)

        tokenizer: Tokenizer = instantiate(cfg.tokenizer).to(device).eval()
        tokenizer.load_state_dict(torch.load(Path(SOURCE_ROOT) / "input_artifacts/tokenizer.pt"))
        action_tokenizer = ActionTokenizer(move_shot_anchors=cfg.env.train.move_shot_anchors)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=action_tokenizer.n_actions,
                                 act_continuous_size=3,
                                 config=instantiate(cfg.world_model),
                                 reward_map=cfg.env.reward_map,
                                 reward_divisor=cfg.env.reward_divisor
                                 ).to(device).eval()
        world_model.load_state_dict(torch.load(Path(SOURCE_ROOT) / "input_artifacts/world_model.pt"))
        cfg.actor_critic.use_original_obs = False
        actor_critic: ConvActorCritic = instantiate(cfg.actor_critic).to(device).train()
        optimizer = torch.optim.Adam(actor_critic.parameters(),
                                     lr=cfg.training.learning_rate)

        train_dataset, eval_dataset = get_dataset_split(cfg)
        imagined_dataset = SegmentDataset(cfg.training.actor_critic.batch_num_samples, max_len=15000)  # ~20GB max
        custom_setup(cfg)

        to_log = {}
        for n_step in tqdm(range(100_000), desc="Steps: "):
            if n_step % 3 == 0 or not imagined_dataset.can_sample:
                batch = train_dataset.sample_batch(cfg.training.actor_critic.batch_num_samples,
                                                   1 + cfg.training.actor_critic.burn_in,
                                                   sample_from_start=False)
                b, t, a = batch["actions"].shape
                tokens = [action_tokenizer.create_action_token(*act.numpy().flatten()) for act in batch["actions"].reshape(-1, a)]
                batch["actions"] = torch.tensor(tokens).reshape(b, t)
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, imaginaition = actor_critic.compute_loss(batch, tokenizer, world_model,
                                                               should_imagine=True,
                                                               **cfg.training.actor_critic)
                imagined_dataset.update(imaginaition, batch)
            else:
                batch = imagined_dataset.sample()
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, _ = actor_critic.compute_loss(batch, tokenizer, world_model,
                                                    should_imagine=False,
                                                    **cfg.training.actor_critic)
            loss.loss_total.backward()

            if (n_step + 1) % cfg.training.actor_critic.grad_acc_steps == 0:
                adaptive_gradient_clipping(actor_critic.parameters(), lam=cfg.training.actor_critic.agc_lambda)
                optimizer.step()
                for p in actor_critic.parameters():
                    p.grad = None

            to_log.update({k: to_log.get(k, []) + [m] for k, m in loss.intermediate_losses.items()})
            if (n_step + 1) % (1*cfg.training.actor_critic.grad_acc_steps) == 0:
                to_log = {f"train/{k}": sum(v)/len(v) for k, v in to_log.items()}
                wandb.log(to_log)
                to_log = {}

            if (n_step + 1) % (100 * 1) == 0:
                torch.save(actor_critic.state_dict(), "checkpoints/last.pt")
                torch.save(optimizer.state_dict(), "checkpoints/optimizer.pt")

    finally:
        wandb.finish()


if __name__ == "__main__":
    # todo: try ssm (mamba 2) to model both wm and actor
    #  https://github.com/state-spaces/mamba/tree/main?tab=readme-ov-file
    # todo: or transformer multitoken prediction https://arxiv.org/pdf/2404.19737
    main()
