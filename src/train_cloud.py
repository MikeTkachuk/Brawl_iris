import os
import sys

sys.path.append(os.getcwd())

import json
import shutil
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
    metrics = trainer.train_agent(trainer.start_epoch)
    with open('/home/ec2-user/checkpoints/metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file)

    trainer.save_checkpoint(trainer.start_epoch, False)
    shutil.copytree(os.getcwd() + r'/checkpoints', r'/home/ec2-user/checkpoints', dirs_exist_ok=True)


if __name__ == "__main__":
    main()
