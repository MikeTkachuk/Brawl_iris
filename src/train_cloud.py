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
# from src.utils import collect_embeddings


def before_epoch(trainer: Trainer, epoch):
    trainer.agent.tokenizer.embedding.reset()

    # # test emb collection
    # if trainer.cfg.training.tokenizer.kmeans_after_epoch < epoch:
    #     embeddings = collect_embeddings(trainer)
    #     file_name = Path(f'embeddings_epoch_{epoch}.pt').name
    #     torch.save(embeddings, str(file_name))
    #     os.system(
    #         f"aws s3 cp {file_name} s3://{trainer.cfg.cloud.bucket_name}/logs/{file_name}")
    #     exit()


def after_epoch(trainer: Trainer, epoch, metrics=None):
    # save tokenizer vocab norms
    tokenizer_vocab_norms_path = Path(trainer.cfg.cloud.log_tokenizer_vocab).name
    vocab_norm = torch.norm(trainer.agent.tokenizer.embedding.weight, dim=-1).detach().cpu()
    with open(tokenizer_vocab_norms_path, 'w') as tok_voc_file:
        json.dump({'step': epoch,
                   'data': {
                       'tokenizer/train/vocab_norms': vocab_norm.numpy().tolist(),
                       'tokenizer/train/vocab_frequency': trainer.agent.tokenizer.embedding._word_stats.counts.detach().cpu().numpy().tolist(),
                           },
                   },
                  tok_voc_file)

    norm_vs_occurrence = torch.stack([vocab_norm, trainer.agent.tokenizer.embedding._word_stats.counts], dim=0)
    norm_occurrence_correlation = torch.corrcoef(norm_vs_occurrence)[0, 1]
    metrics[0]['tokenizer/train/norm_occurrence_correlation'] = norm_occurrence_correlation.item()

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
    os.system(
        f"aws s3 cp {tokenizer_vocab_norms_path} s3://{trainer.cfg.cloud.bucket_name}/{trainer.cfg.cloud.log_tokenizer_vocab}")
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
        print(f"Cloud epoch: {epoch}/{trainer.start_epoch + cfg.training.epochs_per_job}")

        metrics = trainer.train_agent(epoch)

        after_epoch(trainer, epoch, metrics=metrics)


if __name__ == "__main__":
    main()
