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
import matplotlib.pyplot as plt

plt.switch_backend("AGG")

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
    # todo: increase loss contrast (abs(l)^3 / 3 loss)
    # todo: (?) add different maps. tok overfits to patch distribution of single map
    # todo: fit -> add mistakes to dataset -> fit again with new weights
    # todo: [WM] eval with decoder, log tok losses
    # todo: [WM] sequence gan loss for distribution matching
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

        def log_routine(loss_dict: Union[dict, list], mode="train", media=True, batch=None, count_tokens=None):
            if isinstance(loss_dict, list):
                loss_dict = {k: sum([m[k] for m in loss_dict]) / len(loss_dict) for k in loss_dict[0]}
            loss_dict = {f"{mode}/{k}": v for k, v in loss_dict.items()}

            # logs
            to_log = {}
            if media:
                obs_item = batch["observations"][0]
                was_training = tokenizer.training
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
                if was_training:
                    tokenizer.train()
            to_log.update(loss_dict)

            # tokens
            if count_tokens is not None:
                all_tokens_t = torch.cat(count_tokens)
                unique, counts = torch.unique(all_tokens_t, return_counts=True)
                counts = sorted(counts, reverse=True)
                s = []
                for i, c in enumerate(counts):
                    s.append(c.item())
                f, ax = plt.subplots()
                ax.plot(torch.arange(len(s)) / tokenizer.vocab_size, torch.log(torch.tensor(s)))
                to_log[f"{mode}/tokens"] = wandb.Image(f)
                plt.close("all")

            wandb.log(to_log)

        all_tokens = []
        for epoch in range(6):
            data_iterator = iter(dataloader)
            for i in tqdm(range(len(dataloader)), total=len(dataloader), desc=f"Epoch {epoch}: "):
                if i % 400 == 0:
                    tokenizer.eval()
                    eval_loader = eval_dataset.torch_dataloader(cfg.training.tokenizer.batch_num_samples,
                                                                random_sampling=True,
                                                                pin_memory=True,
                                                                tokenizer=True,
                                                                parallelize=False, )
                    eval_iterator = iter(eval_loader)
                    eval_metrics = []
                    eval_tokens = []
                    with torch.no_grad():
                        for eval_i in range(100):
                            sample = next(eval_iterator)
                            batch, op_flow = sample[:2]
                            batch["observations"] = batch["observations"].to(device)
                            tokens = [None]
                            loss = tokenizer.compute_loss(batch, tokens_placeholder=tokens, pixel_weights=None)
                            if tokens[0] is not None:
                                eval_tokens.append(tokens[0].detach())
                            eval_metrics.append(loss.intermediate_losses)
                    log_routine(eval_metrics, mode="eval", batch=batch, count_tokens=eval_tokens)
                    tokenizer.train()

                if i % 1200 == 0:
                    if all_tokens:
                        log_routine({}, media=False, count_tokens=all_tokens)
                        all_tokens = []
                    tokenizer.eval()
                    tokenizer.init_embedding_kmeans(dataloader, num_batches=256)
                    tokenizer.train()
                    # optimizer = torch.optim.Adam(tokenizer.parameters(),
                    #                              lr=cfg.training.learning_rate)

                sample = next(data_iterator)
                batch, op_flow = sample[:2]
                batch["observations"] = batch["observations"].to(device)
                tokens = [None]
                loss = tokenizer.compute_loss(batch, tokens_placeholder=tokens, pixel_weights=None)
                tokenizer.do_backward(loss.loss_total)
                if tokens[0] is not None:
                    all_tokens.append(tokens[0].detach())

                if (i + 1) % cfg.training.tokenizer.grad_acc_steps == 0:
                    # # todo:
                    # g = tokenizer.ad_loss.discriminator[0].weight.grad.abs().mean(0)
                    # f, ax = plt.subplots()
                    # ax.plot(g.cpu().numpy())
                    # wandb.log({"train/discr_grad": wandb.Image(f)})
                    # plt.close("all")
                    # # end todo
                    adaptive_gradient_clipping(tokenizer.parameters(), lam=cfg.training.actor_critic.agc_lambda)
                    optimizer.step()
                    for param in tokenizer.parameters():
                        param.grad = None
                    should_log_rec = (i + 1) % (3 * cfg.training.tokenizer.grad_acc_steps) == 0
                    log_routine(loss.intermediate_losses, media=should_log_rec, batch=batch)

                if (i + 1) % 400 == 0:
                    torch.save(tokenizer.state_dict(), "checkpoints/last.pt")
                    torch.save(optimizer.state_dict(), "checkpoints/optimizer.pt")
    finally:
        # shutil.rmtree(Path(SOURCE_ROOT) / r"input_artifacts/dataset/preprocessed")
        wandb.finish()


if __name__ == "__main__":
    main()
