import shutil
import hydra
import torch
from omegaconf import DictConfig

from src.trainer import Trainer

"""
vocab size 4096 does not degrade latency
tokenizer is the bottleneck in wm training (50% time) and mild AC (10-20%)
wm takes 60%, decoder 15%, imagine reset takes 25%
order of things to do:
- try 16 tokens per image with big vocab
- find optimal tokens vs vocab vs latency for imagination optimization
- tune tokenizer latency (e.g. channels count)
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

        cfg.tokenizer.decoder.config.z_channels = 256
        cfg.tokenizer.decoder.config.ch = 48
        cfg.tokenizer.decoder.config.ch_mult = [1, 2, 4, 4, 5]
        cfg.tokenizer.decoder.config.num_res_blocks = 1

        cfg.world_model.num_layers = 10
        cfg.world_model.embed_dim = 256
        cfg.world_model.num_heads = 4
        cfg.world_model.tokens_per_block = 37

    def configure_for_benchmark():
        cfg.training.tokenizer.steps_per_epoch = 4
        cfg.training.world_model.steps_per_epoch = 4
        cfg.training.actor_critic.steps_per_epoch = 4
        cfg.training.tokenizer.start_after_epochs = 0
        cfg.training.world_model.start_after_epochs = 1000
        cfg.training.actor_critic.start_after_epochs = 1000
        cfg.training.tokenizer.batch_num_samples = 2
        cfg.training.world_model.batch_num_samples = 1
        cfg.training.actor_critic.batch_num_samples = 1

    configure_for_benchmark()
    trainer = Trainer(cfg, cloud_instance=True)
    trainer.train_dataset.load_disk_checkpoint(trainer.ckpt_dir / 'dataset')
    trainer.agent.load(r"C:\Users\Mykhailo_Tkachuk\Downloads\last.pt", device='cpu')

    batch = trainer.train_dataset.sample_batch(1, 1)
    obs = batch['observations'][0].to(trainer.device)
    with torch.no_grad():
        reconstruction_q = trainer.agent.tokenizer.encode_decode(obs, True, True).detach()
        reconstruction = trainer.agent.tokenizer(obs, True, True)[-1].detach()
    reconstruction = torch.cat([obs, reconstruction, reconstruction_q], -2).cpu().mul(255).clamp(0, 255).to(torch.uint8)
    print(reconstruction.shape)
    reconstruction = torch.nn.functional.interpolate(reconstruction, scale_factor=2.0)
    print(reconstruction.shape)
    exit()

    import matplotlib.pyplot as plt
    import numpy as np

    def decode(tokens, feature_size=(6,6)):
        assert len(tokens) == np.prod(feature_size)
        embs = trainer.agent.tokenizer.embedding.weight[tokens].T.reshape(1,256, *feature_size)
        rec = trainer.agent.tokenizer.decode(embs, True).detach().numpy()[0].transpose(1, 2, 0)
        return rec

    def encode(img):
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img.transpose(2,0,1)).unsqueeze(0)
        tokens = trainer.agent.tokenizer.encode(img, True).tokens
        return tokens

    batch = trainer.train_dataset.sample_batch(2,2)
    obs = batch['observations']
    brawl_tokens = [ 636,  636, 1191, 1209,  241,  593, 1252, 1191,  593, 1209,  593,  684,
         684, 1191,  636,  636,  636, 1084, 1252, 1434, 1434, 1434, 1084,  684,
        1209, 1434, 1252,  684, 1209,  684,  636, 1434,  636, 1434, 1084, 1084]
    map_tokens = [1084,  636, 1434, 1209, 1434,  684, 1434, 1209,  241,  636, 1252, 1434,
         636,  684, 1252, 1191,  636, 1084, 1191,  636,  241,  593, 1434,  684,
         684,  636, 1434, 1209,  593, 1252, 1434, 1084,  241,  636,  636, 1434]
    rec = decode([1084,636,684,1434], (2,2))
    plt.imshow(rec)
    plt.show()
    for b in range(1):
        for t in range(2):
            plt.figure()
            tokens = map_tokens
            tokens[t*18:(t+1)*18] = brawl_tokens[t*18:(t+1)*18]
            rec = decode(tokens)
            plt.figure()
            plt.imshow(rec)
    plt.show()

if __name__ == "__main__":
    main()
