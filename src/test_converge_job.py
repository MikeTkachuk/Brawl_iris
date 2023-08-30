import copy
import os
import time
from pathlib import Path
from threading import Thread, Event

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import boto3

from src.aws.compute_instance import InstanceContext
from src.trainer import Trainer
from src.aws.job_runner import JobRunner
from src.aws.logger import LogListener
from src.trainer import log_metrics

DATA_PREFIX = "pretrain_data/eve/solo_showdown"


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
    trainer.log_listeners = [logger_metrics, ]
    trainer.prepare_job()
    commands = [
        "rm -r Brawl_iris.zip Brawl_iris checkpoints",

        f"aws s3 cp \"s3://{cfg.cloud.bucket_name}/{run_prefix}\" ~ --recursive --quiet",
        f"unzip -q Brawl_iris.zip -d Brawl_iris",
        f"sh {repo_root.name}/src/aws/test_converge.sh {run_prefix}",
    ]

    job_runner = JobRunner(cfg.cloud.bucket_name,
                           str(run_prefix),
                           cfg.cloud.instance_id,
                           cfg.cloud.region_name,
                           cfg.cloud.key_file,
                           commands,
                           trainer.log_listeners,
                           )

    def gpu_stats_thread_func(stop_key: Event):
        while True:
            if stop_key.is_set():
                return
            try:
                instance_context = InstanceContext(cfg.cloud.instance_id, cfg.cloud.region_name)
                instance_context.connect(cfg.cloud.key_file)
                break
            except Exception as e:
                print(e)
                seconds_to_sleep = 20
                print(f"Async connection: Could not connect to instance at this time, retrying in {seconds_to_sleep} seconds")
                time.sleep(seconds_to_sleep)

        for i in range(50):
            if stop_key.is_set():
                return
            instance_context.exec_command(
                "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used "
                "--format=csv", quiet=True)
            time.sleep(5)

    gpu_thread_handle = Event()
    gpu_thread = Thread(target=gpu_stats_thread_func, args=(gpu_thread_handle,))
    gpu_thread.start()

    job_runner.run()
    gpu_thread_handle.set()
    gpu_thread.join()

    wandb.finish()


if __name__ == "__main__":
    main()
