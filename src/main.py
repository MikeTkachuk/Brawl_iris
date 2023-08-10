import shutil
import hydra
from omegaconf import DictConfig

from trainer import Trainer

"""
"""


@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    shutil.copytree(r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\outputs\2023-07-15\19-41-49\checkpoints",
                    "checkpoints", dirs_exist_ok=True)
    should_configure = False

    if should_configure:
        cfg.tokenizer.vocab_size = 8192
        cfg.tokenizer.embed_dim = 256

        cfg.tokenizer.encoder.config.resolution = 192
        cfg.datasets.train.resolution = 192

        cfg.tokenizer.encoder.config.z_channels = 256
        cfg.tokenizer.encoder.config.ch = 16
        cfg.tokenizer.encoder.config.ch_mult = [1, 2, 2, 3, 4]
        cfg.tokenizer.encoder.config.num_res_blocks = 1

        cfg.world_model.num_layers = 10
        cfg.world_model.embed_dim = 256
        cfg.world_model.num_heads = 4
        cfg.world_model.tokens_per_block = 17

    def configure_for_benchmark():
        cfg.training.tokenizer.steps_per_epoch = 4
        cfg.training.world_model.steps_per_epoch = 4
        cfg.training.actor_critic.steps_per_epoch = 4
        cfg.training.tokenizer.start_after_epochs = 1000
        cfg.training.world_model.start_after_epochs = 1000
        cfg.training.actor_critic.start_after_epochs = 0
        cfg.training.tokenizer.batch_num_samples = 1
        cfg.training.world_model.batch_num_samples = 1
        cfg.training.actor_critic.batch_num_samples = 2
        cfg.actor_critic.use_original_obs = True

    configure_for_benchmark()
    trainer = Trainer(cfg, cloud_instance=True)
    trainer.train_dataset.load_disk_checkpoint(trainer.ckpt_dir / 'dataset')
    trainer.agent.load(r"C:\Users\Mykhailo_Tkachuk\Downloads\last.pt", 'cpu')
    for i in range(1, 20):
        trainer.train_agent(i)


if __name__ == "__main__":
    main()
