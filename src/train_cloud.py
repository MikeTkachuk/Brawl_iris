import os
import sys

sys.path.append(os.getcwd())

import json
import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.trainer import Trainer


@hydra.main(config_path=r"../config", config_name="trainer")
def main(cfg: DictConfig):
    shutil.copytree(r'/home/ec2-user/checkpoints', os.getcwd() + r'/checkpoints')
    trainer = Trainer(cfg, cloud_instance=True, env_actions=json.loads(str(cfg.env_actions)))
    trainer.load_checkpoint()
    trainer.start_epoch -= 1  # negates increment in load_checkpoint()

    # train code
    for epoch in range(trainer.start_epoch, trainer.start_epoch + cfg.training.epochs_per_job):
        metrics = trainer.train_agent(epoch)
        metrics_file_path = Path(cfg.cloud.log_path).name
        with open(metrics_file_path, 'w') as metrics_file:
            json.dump(metrics, metrics_file)
        os.system(f"aws s3 cp {metrics_file_path} s3://{cfg.cloud.bucket_name}/{cfg.cloud.log_path}")

    trainer.save_checkpoint(trainer.start_epoch + cfg.training.epochs_per_job - 1, save_agent_only=False)
    shutil.copytree(os.getcwd() + r'/checkpoints', r'/home/ec2-user/checkpoints', dirs_exist_ok=True)


if __name__ == "__main__":
    main()
