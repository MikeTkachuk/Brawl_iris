import shutil
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parents[2]))

import hydra
import wandb
from hydra.utils import instantiate

import torch
import torch.backends
from omegaconf import OmegaConf
from tqdm import tqdm

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

        shutil.copytree(
            Path(
                r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\log_eval\2024-04-25_08-17-09\checkpoints\dataset"),
            Path("checkpoints\dataset"), dirs_exist_ok=True, )
        dataset.load_disk_checkpoint(Path("checkpoints/dataset"))
        dataloader = dataset.torch_dataloader(cfg.training.tokenizer.batch_num_samples,
                                              random_sampling=False,
                                              pin_memory=True,
                                              tokenizer=True,
                                              num_workers=2,
                                              prefetch_factor=20)

        def temperature_schedule(step, from_t=1.0, to_t=0.5, decay_steps=3758):
            if step < decay_steps:
                return from_t + step / decay_steps * (to_t - from_t)
            return to_t

        custom_setup(cfg)

        all_tokens = []
        for epoch in range(6):
            tokenizer.init_embedding_kmeans(dataloader)
            data_iterator = iter(dataloader)
            for i in tqdm(range(len(dataloader)), total=len(dataloader), desc=f"Epoch {epoch}: "):
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
                    tokenizer.embedding.temperature = temperature_schedule(epoch * len(dataloader) + i - 1)

                    # logs
                    to_log = {}
                    if (i + 1) % (3 * cfg.training.tokenizer.grad_acc_steps) == 0:
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
                        to_log["reconstructions"] = wandb.Image(reconstruction[0].permute(1, 2, 0).numpy())
                    to_log.update(loss.intermediate_losses)
                    to_log["gumbel_temperature"] = tokenizer.embedding.temperature

                    # tokens
                    if all_tokens:
                        all_tokens = torch.cat(all_tokens)
                        unique, counts = torch.unique(all_tokens, return_counts=True)
                        counts = sorted(counts, reverse=True)
                        s = 0
                        evenness_metric = 0
                        thresh = 0.95
                        for i, c in enumerate(counts):
                            s += c / torch.numel(all_tokens)
                            if s > thresh:
                                evenness_metric = i / tokenizer.vocab_size
                                break
                        to_log[f"evenness_{thresh}"] = evenness_metric

                        all_tokens = []

                    wandb.log(to_log)
                    tokenizer.train()

                if (i + 1) % 50 == 0:
                    torch.save(tokenizer.state_dict(), "checkpoints/last.pt")
                    torch.save(optimizer.state_dict(), "checkpoints/optimizer.pt")
    finally:
        shutil.rmtree(r"checkpoints/dataset")
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
