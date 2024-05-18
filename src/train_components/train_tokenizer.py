import shutil
import sys
from collections import deque
from pathlib import Path
import time
from typing import Union

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
from src.utils import set_seed, adaptive_gradient_clipping
from src.models.tokenizer.tokenizer import Tokenizer

device = "cuda"


def custom_setup(cfg):
    torch.backends.cudnn.benchmark = True
    set_seed(cfg.common.seed)
    if sys.gettrace() is not None:  # if debugging
        cfg.wandb.mode = "offline"
    cfg.wandb.tags = list(set(cfg.wandb.tags or [] + ["tokenizer"]))
    wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
               reinit=True,
               resume=True,
               **cfg.wandb)


@hydra.main(config_path="../../config", config_name="trainer")
def main(cfg):
    try:
        Path("checkpoints/dataset").mkdir(parents=True, exist_ok=True)

        dataset: EpisodesDataset = instantiate(cfg.datasets.train)
        tokenizer: Tokenizer = instantiate(cfg.tokenizer).to(device).train()
        optimizer = torch.optim.Adam(tokenizer.parameters(),
                                     lr=cfg.training.learning_rate)

        dataset.load_disk_checkpoint(Path(SOURCE_ROOT) / "input_artifacts/dataset")
        episode_ids = dataset.disk_episodes
        split = train_test_split(episode_ids, test_size=0.1)
        train_dataset: EpisodesDataset = instantiate(cfg.datasets.train)
        train_dataset.disk_episodes = deque(split[0])
        eval_dataset: EpisodesDataset = instantiate(cfg.datasets.train)
        eval_dataset.disk_episodes = deque(split[1])

        train_dataset._dir = eval_dataset._dir = dataset._dir

        dataloader = train_dataset.torch_dataloader(cfg.training.tokenizer.batch_num_samples,
                                              random_sampling=False,
                                              pin_memory=True,
                                              tokenizer=True,
                                              num_workers=2,
                                              prefetch_factor=4)

        custom_setup(cfg)

        all_tokens = []

        def log_routine(loss_dict: Union[dict, list], mode="train", media=True, batch=None):
            if isinstance(loss_dict, list):
                loss_dict = {k: sum([m[k] for m in loss_dict]) / len(loss_dict) for k in loss_dict[0]}
            loss_dict = {f"{mode}/{k}": v for k, v in loss_dict.items()}

            # logs
            to_log = {}
            if media:
                obs_item = batch["observations"][0]
                tokenizer.eval()
                with torch.no_grad():
                    reconstruction_q = tokenizer.encode_decode(obs_item, True, True).detach()
                    reconstruction = tokenizer.encode_decode(obs_item, True, True, False).detach()
                reconstruction = torch.cat([obs_item, reconstruction_q, reconstruction], -2).cpu().mul(
                    255).clamp(0,
                               255).to(
                    torch.uint8)
                reconstruction = torch.nn.functional.interpolate(reconstruction, scale_factor=(1.0, 2.0))
                to_log[f"{mode}/reconstructions"] = wandb.Image(reconstruction[0].permute(1, 2, 0).numpy())
            to_log.update(loss_dict)

            # tokens
            if all_tokens:
                all_tokens_t = torch.cat(all_tokens)
                unique, counts = torch.unique(all_tokens_t, return_counts=True)
                counts = sorted(counts, reverse=True)
                s = 0
                evenness_metric = 0
                thresh = 0.95
                for i, c in enumerate(counts):
                    s += c / torch.numel(all_tokens_t)
                    if s > thresh:
                        evenness_metric = i / tokenizer.vocab_size
                        break
                to_log[f"{mode}/evenness_{thresh}"] = evenness_metric

                all_tokens.clear()

            wandb.log(to_log)
            tokenizer.train()

        for epoch in range(6):
            data_iterator = iter(dataloader)
            for i in tqdm(range(len(dataloader)), total=len(dataloader), desc=f"Epoch {epoch}: "):
                if i % 400 == 0:
                    tokenizer.eval()
                    eval_loader = eval_dataset.torch_dataloader(cfg.training.tokenizer.batch_num_samples,
                                              random_sampling=True,
                                              pin_memory=True,
                                              tokenizer=True,
                                              parallelize=False,)
                    eval_iterator = iter(eval_loader)
                    eval_metrics = []
                    all_tokens = []
                    with torch.no_grad():
                        for eval_i in range(50):
                            sample = next(eval_iterator)
                            batch, op_flow = sample[:2]
                            batch["observations"] = batch["observations"].to(device)
                            tokens = [None]
                            loss = tokenizer.compute_loss(batch, tokens_placeholder=tokens, pixel_weights=None)
                            if tokens[0] is not None:
                                all_tokens.append(tokens[0].detach())
                            eval_metrics.append(loss.intermediate_losses)
                    log_routine(eval_metrics, mode="eval", batch=batch)
                    tokenizer.train()

                if i % 1200 == 0:
                    tokenizer.init_embedding_kmeans(dataloader)
                    optimizer = torch.optim.Adam(tokenizer.parameters(),
                                                 lr=cfg.training.learning_rate)

                sample = next(data_iterator)
                batch, op_flow = sample[:2]
                batch["observations"] = batch["observations"].to(device)
                tokens = [None]
                loss = tokenizer.compute_loss(batch, tokens_placeholder=tokens, pixel_weights=None)
                loss.loss_total.backward()
                if tokens[0] is not None:
                    all_tokens.append(tokens[0].detach())

                if (i + 1) % cfg.training.tokenizer.grad_acc_steps == 0:
                    adaptive_gradient_clipping(tokenizer.parameters(), lam=cfg.training.actor_critic.agc_lambda)
                    optimizer.step()
                    for param in tokenizer.parameters():
                        param.grad = None
                    should_log_rec = (i+1)%(3*cfg.training.tokenizer.grad_acc_steps)==0
                    log_routine(loss.intermediate_losses, media=should_log_rec, batch=batch)


                if (i + 1) % 400 == 0:
                    torch.save(tokenizer.state_dict(), "checkpoints/last.pt")
                    torch.save(optimizer.state_dict(), "checkpoints/optimizer.pt")
    finally:
        # shutil.rmtree(Path(SOURCE_ROOT) / r"input_artifacts/dataset/preprocessed")
        wandb.finish()


def inspect_decoder():
    with hydra.initialize(config_path="../config"):
        cfg = hydra.compose(config_name="trainer")

    tokenizer: Tokenizer = instantiate(cfg.tokenizer)

    def assemble_z_q(tokens):
        tokens = torch.tensor(tokens, dtype=torch.long).reshape(1, 8, 8)
        z_q = tokenizer.embedding.weight[tokens]
        return z_q

    tokenizer.load_state_dict(
        torch.load(r"C:\Users\Michael\PycharmProjects\Brawl_iris\outputs\dist_quant_again\2024-04-30_23-43-51\checkpoints\last.pt")
    ).to(device)
    tokenizer.eval()
    episode = torch.load(Path(
                r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\log_eval\2024-04-25_08-17-09\checkpoints\dataset\32.pt"),
            )
    frame = episode["observations"][12] / 255.0

    enc_output = tokenizer.encode(frame[None, None], should_preprocess=True)
    decoded = tokenizer.decode(enc_output.z_quantized, should_postprocess=True)


if __name__ == "__main__":
    main()
