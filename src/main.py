import shutil
import hydra
import torch
import torchvision.io
from omegaconf import DictConfig
from torchvision.io import ImageReadMode

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
        cfg.training.world_model.start_after_epochs = 0
        cfg.training.actor_critic.start_after_epochs = 0
        cfg.training.tokenizer.batch_num_samples = 1
        cfg.training.world_model.batch_num_samples = 1
        cfg.training.actor_critic.batch_num_samples = 1

    configure_for_benchmark()
    trainer = Trainer(cfg, cloud_instance=True)
    trainer.train_dataset.load_disk_checkpoint(trainer.ckpt_dir / 'dataset')
    # trainer.agent.load(r"C:\Users\Mykhailo_Tkachuk\Downloads\last.pt", device='cpu')

    for epoch in range(1, 2):
        trainer.train_agent(epoch)
    exit()

    import matplotlib.pyplot as plt
    import numpy as np

    def tok_2_emb(tokens):
        return trainer.agent.tokenizer.embedding.weight[tokens]

    def decode(tokens, feature_size=(6, 6)):
        tokens = tokens.flatten()
        assert len(tokens) == np.prod(feature_size)
        embs = trainer.agent.tokenizer.embedding.weight[tokens].T.reshape(1, 256, *feature_size)
        rec = trainer.agent.tokenizer.decode(embs, True)
        return rec

    def encode(img):
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0)
        tokens = trainer.agent.tokenizer.encode(img, True).tokens
        print(tokens)
        return tokens

    def plot(img):
        if len(img.shape) > 3:
            img = img.reshape(img.shape[-3:])
        if isinstance(img, torch.Tensor):
            img = img.detach().numpy().transpose(1, 2, 0)

        plt.imshow(img)

    def token_cosine(token1, token2):
        emb1 = trainer.agent.tokenizer.embedding.weight[token1]
        emb2 = trainer.agent.tokenizer.embedding.weight[token2]
        return torch.sum(emb1 * emb2, dim=1) / torch.norm(emb1, dim=1) / torch.norm(emb2, dim=1)

    for i in range(4):
        img = torchvision.io.read_image(fr"C:\Users\Mykhailo_Tkachuk\Downloads\reconstruction{i}_inpainted.png", ImageReadMode.RGB)[:, :192,
              :192] / 255
        img_orig = torchvision.io.read_image(fr"C:\Users\Mykhailo_Tkachuk\Downloads\reconstruction{i}.png", ImageReadMode.RGB)[:, :192,
                   :192] / 255

        img_tokens = encode(img.unsqueeze(0))
        img_orig_tokens = encode(img_orig.unsqueeze(0))
        mask = img_tokens != img_orig_tokens
        mask_img = torch.repeat_interleave(mask.reshape(6, 6), 32, dim=1)
        mask_img = torch.repeat_interleave(mask_img, 32, dim=0) / 2 + 0.5
        plot(decode(img_tokens) * mask_img)
        plt.figure(), plot(decode(img_orig_tokens) * mask_img)
        print(img_tokens == img_orig_tokens)
        print(img_tokens[mask])
        print(img_orig_tokens[mask])
        print(token_cosine(img_tokens[mask], img_orig_tokens[mask]))
        print(tok_2_emb(img_tokens[mask])[:,:10])
        print(tok_2_emb(img_orig_tokens[mask])[:,:10])
        plt.show()


if __name__ == "__main__":
    main()
