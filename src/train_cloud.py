import os
import random
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


@hydra.main(config_path=r"../config", config_name="trainer")
def main(cfg: DictConfig):
    shutil.copytree(r'/home/ec2-user/checkpoints', os.getcwd() + r'/checkpoints')
    trainer = Trainer(cfg, cloud_instance=True)
    trainer.run_prefix = cfg.run_prefix
    trainer.load_checkpoint()
    trainer.start_epoch -= 1  # negates increment in load_checkpoint()

    # train code
    for epoch in range(trainer.start_epoch, trainer.start_epoch + cfg.training.epochs_per_job):
        print(f"Cloud epoch: {epoch}/{trainer.start_epoch + cfg.training.epochs_per_job}")

        metrics = trainer.train_agent(epoch)

        # save metrics
        metrics_file_path = Path(cfg.cloud.log_metrics).name
        with open(metrics_file_path, 'w') as metrics_file:
            json.dump(metrics, metrics_file)

        # save reconstructions
        reconstruction_path = Path(cfg.cloud.log_reconstruction).name
        episode = trainer.train_dataset.get_episode(random.randint(0, len(trainer.train_dataset)-1))
        obs = episode.observations[int(len(episode) / random.uniform(1.2, 4))].to(trainer.device).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            reconstruction = trainer.agent.tokenizer.encode_decode(obs, True, True)
        reconstruction = reconstruction.detach()
        reconstruction = torch.cat([obs, reconstruction], -1).cpu().mul(255).clamp(0, 255).to(torch.uint8)[0]
        reconstruction = torch.nn.functional.interpolate(reconstruction, scale_factor=2.5)
        write_jpeg(reconstruction, str(reconstruction_path))

        os.system(f"aws s3 cp {reconstruction_path} s3://{cfg.cloud.bucket_name}/{cfg.cloud.log_reconstruction}")
        os.system(f"aws s3 cp {metrics_file_path} s3://{cfg.cloud.bucket_name}/{cfg.cloud.log_metrics}")
        trainer.save_checkpoint(epoch, save_agent_only=False, save_dataset=False)
        os.system(f'aws s3 cp checkpoints s3://brawl-stars-iris/{trainer.run_prefix}/checkpoints '
                  f'--recursive '
                  f'--exclude "dataset/*"')


if __name__ == "__main__":
    main()
