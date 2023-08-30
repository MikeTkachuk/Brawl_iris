import os
from pathlib import Path
import shutil

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import boto3

from src.trainer import Trainer
from src.aws.job_runner import JobRunner
from src.aws.logger import LogListener
from src.trainer import log_metrics, log_image, log_histogram

DATA_PREFIX = "pretrain_data/eve/solo_showdown"


def init_dataset():
    dst = Path("/home/ec2-user/checkpoints/dataset")
    dst.mkdir(parents=True, exist_ok=False)
    for i, file in enumerate((Path("/home/ec2-user") / DATA_PREFIX).rglob("*.pt")):
        file.rename(dst / f"{i}.pt")
    shutil.rmtree(Path("/home/ec2-user") / Path(DATA_PREFIX).parts[0])


@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        resume=True,
        **cfg.wandb
    )
    run_prefix = Path('_'.join([cfg.wandb.name, Path(os.getcwd()).parent.name, Path(os.getcwd()).name]))
    trainer = Trainer(cfg, cloud_instance=True)
    trainer.run_prefix = run_prefix
    repo_root = Path(__file__).parents[1]  # ->Brawl_iris/src/pretraining.py

    trainer.from_pretrained()
    if cfg.common.resume:
        trainer.resume_run()

    trainer.save_checkpoint(trainer.start_epoch, save_agent_only=False)

    s3_client = boto3.client('s3')
    logger_metrics = LogListener(log_metrics, cfg.cloud.log_metrics, cfg.cloud.bucket_name, s3_client)
    logger_reconstructions = LogListener(log_image,
                                         cfg.cloud.log_reconstruction,
                                         cfg.cloud.bucket_name,
                                         s3_client,
                                         'reconstructions')
    trainer.log_listeners = [logger_metrics, logger_reconstructions]
    trainer.prepare_job()
    commands = [
        "rm -r Brawl_iris checkpoints",
        f"aws s3 cp \"s3://{cfg.cloud.bucket_name}/{DATA_PREFIX}\" /home/ec2-user/{DATA_PREFIX} --recursive --quiet",

        f"aws s3 cp \"s3://{cfg.cloud.bucket_name}/{run_prefix}\" ~ --recursive --quiet",

        f"unzip -q Brawl_iris.zip -d Brawl_iris",

        f"sh {repo_root.name}/src/aws/run_pretrain.sh {run_prefix}",
    ]

    job_runner = JobRunner(cfg.cloud.bucket_name,
                           str(run_prefix),
                           cfg.cloud.instance_id,
                           cfg.cloud.region_name,
                           cfg.cloud.key_file,
                           commands,
                           trainer.log_listeners,
                           )
    job_runner.run()

    wandb.finish()


if __name__ == "__main__":
    main()
