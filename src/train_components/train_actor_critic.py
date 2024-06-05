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

import torch
import torch.backends
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.dataset import EpisodesDataset
from src.utils import set_seed, adaptive_gradient_clipping, ActionTokenizer
from src.models.tokenizer.tokenizer import Tokenizer
from src.models.world_model import WorldModel
from src.models.actor_critic import ConvActorCritic

device = "cuda"


class SegmentDataset:
    pass


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
    if sys.gettrace() is not None:  # if debugging
        cfg.wandb.mode = "offline"
        cfg.training.world_model.batch_num_samples = 2

    cfg.wandb.tags = list(set(cfg.wandb.tags or [] + ["world_model"]))
    wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
               reinit=True,
               resume=True,
               **cfg.wandb)


@hydra.main(config_path="../../config", config_name="trainer")
def main(cfg):
    try:
        print(f"PID: {os.getpid()}")
        set_seed(cfg.common.seed)

        Path("checkpoints/dataset").mkdir(parents=True, exist_ok=True)

        tokenizer: Tokenizer = instantiate(cfg.tokenizer).to(device).eval()
        action_tokenizer = ActionTokenizer(move_shot_anchors=cfg.env.train.move_shot_anchors)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=action_tokenizer.n_actions,
                                 act_continuous_size=3,
                                 config=instantiate(cfg.world_model)).to(device).eval()
        cfg.actor_critic.use_original_obs = False
        actor_critic: ConvActorCritic = instantiate(cfg.actor_critic).to(device).train()
        optimizer = torch.optim.Adam(actor_critic.parameters(),
                                     lr=cfg.training.learning_rate)

        train_dataset, eval_dataset = get_dataset_split(cfg)
        custom_setup(cfg)

        for n_step in tqdm(range(100_000), desc="Steps: "):
            batch = train_dataset.sample_batch(cfg.training.actor_critic.batch_num_samples,
                                               1 + cfg.training.actor_critic.burn_in,
                                               sample_from_start=False)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = actor_critic.compute_loss(batch, tokenizer, world_model, **cfg.training.actor_critic)
            loss.loss_total.backward()

            optimizer.step()
            for p in actor_critic.parameters():
                p.grad = None


    finally:
        wandb.finish()


if __name__ == "__main__":
    # todo: try ssm (mamba 2) to model both wm and actor
    #
    main()
