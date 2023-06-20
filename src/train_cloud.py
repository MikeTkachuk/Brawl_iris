import os
import sys

sys.path.append(os.getcwd())

import json
import hydra
from omegaconf import DictConfig

from src.trainer import Trainer


@hydra.main(config_path=r"../config", config_name="trainer")
def main(cfg: DictConfig):
    trainer = Trainer(cfg, cloud_instance=True, env_actions=json.loads(str(cfg.env_actions)))
    trainer.load_checkpoint()
    trainer.start_epoch -= 1  # negates increment in load_checkpoint()
    trainer.save_checkpoint(trainer.start_epoch, False)


if __name__ == "__main__":
    main()
