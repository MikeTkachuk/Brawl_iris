import os
import sys

sys.path.append(os.getcwd())

import json
import shutil
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig
from torchvision.io import write_jpeg

from src.trainer import Trainer


def before_epoch(trainer: Trainer, epoch):
    pass


def after_epoch(trainer: Trainer, epoch, metrics=None):
    # save metrics
    metrics_file_path = Path(trainer.cfg.cloud.log_metrics).name
    with open(metrics_file_path, 'w') as metrics_file:
        json.dump(metrics, metrics_file)

    # save reconstructions
    reconstruction_path = Path(trainer.cfg.cloud.log_reconstruction).name
    batch = trainer.train_dataset.sample_batch(1, 1)
    obs = batch['observations'][0].to(trainer.device)
    with torch.no_grad():
        reconstruction_q = trainer.agent.tokenizer.encode_decode(obs, True, True).detach()
        reconstruction = trainer.agent.tokenizer(obs, True, True)[-1].detach()
    reconstruction = torch.cat([obs, reconstruction, reconstruction_q], -2).cpu().mul(255).clamp(0, 255).to(torch.uint8)
    reconstruction = torch.nn.functional.interpolate(reconstruction, scale_factor=(1.0, 2.0))
    write_jpeg(reconstruction[0], str(reconstruction_path))

    os.system(
        f"aws s3 cp {reconstruction_path} s3://{trainer.cfg.cloud.bucket_name}/{trainer.cfg.cloud.log_reconstruction}")
    os.system(f"aws s3 cp {metrics_file_path} s3://{trainer.cfg.cloud.bucket_name}/{trainer.cfg.cloud.log_metrics}")
    trainer.save_checkpoint(epoch, save_agent_only=False, save_dataset=False)
    os.system(f'aws s3 cp checkpoints s3://brawl-stars-iris/{trainer.run_prefix}/checkpoints '
              f'--recursive '
              f'--exclude "dataset/*"')


@hydra.main(config_path=r"../config", config_name="trainer")
def main(cfg: DictConfig):
    shutil.copytree(r'/home/ec2-user/checkpoints', os.getcwd() + r'/checkpoints')
    trainer = Trainer(cfg, cloud_instance=True)
    trainer.run_prefix = cfg.run_prefix
    trainer.load_checkpoint()
    trainer.start_epoch -= 1  # negates increment in load_checkpoint()

    # train code
    for epoch in range(trainer.start_epoch, trainer.start_epoch + cfg.training.epochs_per_job):
        before_epoch(trainer, epoch)
        print(f"Cloud epoch: {epoch}/{trainer.start_epoch + cfg.training.epochs_per_job - 1}")

        metrics = trainer.train_agent(epoch)

        after_epoch(trainer, epoch, metrics=metrics)


if __name__ == "__main__":
    main()
