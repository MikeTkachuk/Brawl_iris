import math
import random
import shutil
import sys
from collections import deque
from pathlib import Path
import time
import os
from typing import Union

from einops import rearrange
from torch.distributions import Categorical

SOURCE_ROOT = str(Path(__file__).absolute().parents[2])
sys.path.append(SOURCE_ROOT)

import hydra
import wandb
from hydra.utils import instantiate

import torch
import torch.backends
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.dataset import EpisodesDataset
from src.utils import set_seed, adaptive_gradient_clipping, ActionTokenizer, LossWithIntermediateLosses
from src.models.tokenizer.tokenizer import Tokenizer
from src.models.world_model import WorldModel

device = "cuda"


class JointWM(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer: Tokenizer = instantiate(cfg.tokenizer)
        self.action_tokenizer = ActionTokenizer(move_shot_anchors=cfg.env.train.move_shot_anchors)
        self.world_model = WorldModel(self.tokenizer.embed_dim,  # vocab size == emb dim
                                      act_vocab_size=self.action_tokenizer.n_actions,
                                      act_continuous_size=3,
                                      config=instantiate(cfg.world_model),
                                      reward_map=cfg.env.reward_map,
                                      reward_divisor=cfg.env.reward_divisor
                                      )
        self.wm_embed_map = torch.nn.Linear(self.tokenizer.embed_dim, self.world_model.config.embed_dim)

    @staticmethod
    def norm(x, dim=-1, eps=1E-4, effect=1.0):
        normed = x / (torch.norm(x, dim=dim, keepdim=True) + eps) * math.sqrt(x.size(dim))
        return normed * effect + (1 - effect) * x

    def encode_observations(self, x):
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.tokenizer.encoder(x)
        z = self.tokenizer.pre_quant_norm(z)
        z = self.tokenizer.pre_quant_conv(z)
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z = self.norm(z, dim=-3, effect=0.1)
        return z

    def decode_observations(self, x):
        shape = x.shape  # (..., E, h, w)
        x = x.view(-1, *shape[-3:])
        x = self.tokenizer.post_quant_conv(x)
        rec = self.tokenizer.decoder(x)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        return rec

    def wm_prepare_input(self, embeddings, actions, actions_continuous, past_keys_values=None):
        embeddings = self.wm_embed_map(embeddings)
        placeholder_tokens = torch.cat((torch.zeros(embeddings.shape[:-1],
                                                    dtype=actions.dtype,
                                                    device=actions.device), actions[..., None]), dim=2)
        placeholder_tokens = rearrange(placeholder_tokens, "b l k -> b (l k)")
        embeddings = rearrange(embeddings, "b t l e -> b (t l) e")

        num_steps = placeholder_tokens.size(1)  # (B, T)
        assert num_steps <= self.world_model.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size
        sequences = self.world_model.embedder(placeholder_tokens, num_steps, prev_steps, continuous=actions_continuous)
        action_table_id = self.world_model.embedder.action_table_id
        action_mask = torch.zeros_like(placeholder_tokens[0])
        action_mask[self.world_model.embedder.slicers[action_table_id].compute_slice(num_steps, prev_steps)] = 1
        sequences[:, ~action_mask.bool()] = embeddings
        return sequences

    def wm_forward(self, sequences, past_keys_values=None):
        num_steps = sequences.size(1)  # (B, T, ...)
        assert num_steps <= self.world_model.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size
        if sequences.size(-1) != self.world_model.config.embed_dim:
            sequences = self.wm_embed_map(sequences)
        sequences = sequences + self.world_model.pos_emb(prev_steps + torch.arange(num_steps, device=sequences.device))

        x = self.world_model.transformer(sequences, past_keys_values)
        obs_per_head = [head(x, num_steps=num_steps, prev_steps=prev_steps) for head in self.world_model.obs_heads]

        logits_observations = rearrange(obs_per_head, "h b t e -> b (t h) e")
        logits_observations = self.norm(logits_observations, effect=0.1)
        logits_rewards = self.world_model.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = self.world_model.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)
        return logits_observations, logits_rewards, logits_ends

    def compute_update(self, batch, backward=True, viz=False):
        x = self.tokenizer.preprocess_input(batch["observations"])
        mask_padding = batch["mask_padding"]
        x_valid = x[mask_padding]
        tok_samples = torch.randperm(x_valid.size(0))[:self.cfg.training.tokenizer.batch_num_samples]
        x_tok = x_valid[tok_samples]
        emb_tok = self.encode_observations(x_tok)
        rec = self.decode_observations(emb_tok)
        tok_rec_loss = torch.square(x_tok - rec).mean()
        tok_perc_loss = torch.mean(self.tokenizer.lpips(x_tok, rec))
        tok_loss = LossWithIntermediateLosses(
            tok_rec_loss=tok_rec_loss * self.tokenizer.loss_weights[1],
            tok_perc_loss=tok_perc_loss * self.tokenizer.loss_weights[2],
        )
        if backward:
            (tok_loss.loss_total * self.cfg.training.tokenizer_weight).backward()
        emb = self.encode_observations(x)
        emb_flat = rearrange(emb, "b t e h w -> b t (h w) e")
        wm_sequence = self.wm_prepare_input(emb_flat, batch["actions"], batch["actions_continuous"])
        emb_flat2 = rearrange(emb_flat, "b t l e -> b (t l) e")
        logit_obs, logit_r, logit_end = self.wm_forward(wm_sequence)
        pred_emb_flat = torch.cat([emb_flat2[:, :self.world_model.config.n_gen_heads],
                                   logit_obs[:, :-self.world_model.config.n_gen_heads]], dim=1)
        pred_emb = rearrange(pred_emb_flat, "b (t h w) e -> b t e h w", h=emb.size(-2), w=emb.size(-1))
        _, labels_rewards, labels_ends = self.world_model.compute_labels_world_model(
            torch.zeros_like(emb_flat[..., 0]),  # placeholder
            batch['rewards'],
            batch['ends'],
            batch['mask_padding'],
        )

        # todo: emb wm labels detach vs no detach experiment
        wm_obs_loss = torch.subtract(emb.detach()[batch['mask_padding']],
                                     pred_emb[batch['mask_padding']]).abs().mean()
        loss_rewards = torch.nn.functional.cross_entropy(rearrange(logit_r, 'b t e -> (b t) e'), labels_rewards,
                                                         reduction="none")
        loss_rewards = (loss_rewards * (1 - (-loss_rewards).exp()).pow(2)).mean()
        loss_ends = torch.nn.functional.cross_entropy(rearrange(logit_end, 'b t e -> (b t) e'), labels_ends,
                                                      reduction="none")
        loss_ends = (loss_ends * (1 - (-loss_ends).exp()).pow(2)).mean()
        pred_rec = self.decode_observations(pred_emb[mask_padding][tok_samples])
        wm_rec_loss = torch.square(x_tok - pred_rec).mean()
        wm_perc_loss = torch.mean(self.tokenizer.lpips(x_tok, pred_rec))
        wm_loss = LossWithIntermediateLosses(
            wm_obs_loss=4*wm_obs_loss,
            loss_rewards=loss_rewards,
            loss_ends=loss_ends,
            wm_rec_loss=wm_rec_loss * 0.3,
            wm_perc_loss=wm_perc_loss,
        )
        if backward:
            wm_loss.loss_total.backward()

        metrics = {
            **tok_loss.intermediate_losses,
            **wm_loss.intermediate_losses,
        }
        if viz:
            with torch.no_grad():
                orig = x_tok[0]
                tok_rec = rec[0]
                wm_rec = pred_rec[0]
            reconstruction = torch.cat([orig, tok_rec, wm_rec], -2).add(1).div(2).cpu().mul(
                255).clamp(0,
                           255).to(
                torch.uint8)
            reconstruction = torch.nn.functional.interpolate(reconstruction[None], scale_factor=(1.0, 2.0))
            metrics["viz"] = wandb.Image(reconstruction[0].permute(1, 2, 0).numpy())

        return metrics


def custom_setup(cfg):
    torch.backends.cudnn.benchmark = True
    if sys.gettrace() is not None:  # if debugging
        cfg.wandb.mode = "offline"
        cfg.training.world_model.batch_num_samples = 2
        cfg.training.tokenizer.batch_num_samples = 2

    cfg.wandb.tags = list(set(cfg.wandb.tags or [] + ["joint_world_model"]))
    wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
               reinit=True,
               resume=True,
               **cfg.wandb)


def read_only_dataset(dataset: EpisodesDataset) -> EpisodesDataset:
    dataset.add_episode = None
    dataset.update_disk_checkpoint = None
    return dataset


def get_dataset_split(cfg):
    dataset: EpisodesDataset = read_only_dataset(instantiate(cfg.datasets.train))
    dataset.load_disk_checkpoint(Path(SOURCE_ROOT) / "input_artifacts/dataset")
    episode_ids = dataset.disk_episodes
    split = train_test_split(episode_ids, test_size=0.1)
    train_dataset: EpisodesDataset = read_only_dataset(instantiate(cfg.datasets.train))
    train_dataset.disk_episodes = deque(split[0])
    eval_dataset: EpisodesDataset = read_only_dataset(instantiate(cfg.datasets.train))
    eval_dataset.disk_episodes = deque(split[1])

    train_dataset._dir = eval_dataset._dir = dataset._dir
    return train_dataset, eval_dataset


@hydra.main(config_path="../../config", config_name="joint")
def main(cfg):
    try:
        print(f"PID: {os.getpid()}")
        set_seed(cfg.common.seed)

        Path("checkpoints/dataset").mkdir(parents=True, exist_ok=True)

        joint_wm = JointWM(cfg).to(device).train()
        optimizer = torch.optim.Adam(joint_wm.parameters(),
                                     lr=cfg.training.learning_rate,
                                     weight_decay=cfg.training.world_model.weight_decay)

        # load checkpoint
        state_dict = torch.load(r"C:\Users\Michael\PycharmProjects\Brawl_iris\outputs\norm\2024-06-22_14-47-24\checkpoints\last.pt")
        joint_wm.load_state_dict(state_dict)
        # optimizer.load_state_dict(torch.load(r"C:\Users\Michael\PycharmProjects\Brawl_iris\outputs\more_reg\2024-05-25_18-41-56\checkpoints\optimizer.pt", map_location=device))

        custom_setup(cfg)
        train_dataset, eval_dataset = get_dataset_split(cfg)

        def aggregate_metrics(mt: list, mode="train"):
            out = {}
            for m_name in mt[0]:
                if isinstance(mt[0][m_name], torch.Tensor):
                    out[f"{mode}/{m_name}"] = sum([m[m_name] for m in mt]) / len(mt)
                else:
                    out[f"{mode}/{m_name}"] = mt[0][m_name]
            return out

        to_log = []
        for n_step in tqdm(range(100_000), desc="Steps: "):
            batch = train_dataset.sample_batch(
                max(cfg.training.tokenizer.batch_num_samples // cfg.common.sequence_length,
                    cfg.training.world_model.batch_num_samples),
                cfg.common.sequence_length,
                sample_from_start=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            metrics = joint_wm.compute_update(batch, viz=n_step % 10 == 0)
            to_log.append(metrics)
            if (n_step + 1) % cfg.training.grad_acc_steps == 0:
                adaptive_gradient_clipping(joint_wm.parameters(), lam=cfg.training.world_model.agc_lambda)
                optimizer.step()
                for p in joint_wm.parameters():
                    p.grad = None

                if (n_step + 1) % (cfg.training.grad_acc_steps * 4) == 0:
                    wandb.log(aggregate_metrics(to_log, "train"))
                    to_log = []

            if n_step % (500 * cfg.training.grad_acc_steps) == 0:
                joint_wm.eval()
                eval_metrics = []
                with torch.no_grad():
                    for i in range(100):
                        batch = eval_dataset.sample_batch(
                            max(cfg.training.tokenizer.batch_num_samples // cfg.common.sequence_length,
                                cfg.training.world_model.batch_num_samples),
                            cfg.common.sequence_length,
                            sample_from_start=True)
                        batch = {k: v.to(device) for k, v in batch.items()}
                        eval_metrics.append(joint_wm.compute_update(batch, False, viz=i == 0))
                avg_eval_metrics = aggregate_metrics(eval_metrics, mode="eval")
                wandb.log(avg_eval_metrics)
                joint_wm.train()

            if (n_step + 1) % (100 * cfg.training.grad_acc_steps) == 0:
                torch.save(joint_wm.state_dict(), "checkpoints/last.pt")
                torch.save(optimizer.state_dict(), "checkpoints/optimizer.pt")
            if (n_step + 1) % (4000 * cfg.training.grad_acc_steps) == 0:
                Path(f"checkpoints{n_step // 1000}").mkdir()
                torch.save(joint_wm.state_dict(), f"checkpoints{n_step // 1000}/last.pt")
                torch.save(optimizer.state_dict(), f"checkpoints{n_step // 1000}/optimizer.pt")

    finally:
        wandb.finish()


class ContWMEnv:
    def __init__(self, module: JointWM) -> None:

        self.device = torch.device(device)
        self.module = module
        self.world_model = module.world_model.to(self.device).eval()
        self.tokenizer = module.tokenizer.to(self.device).eval()

        self.keys_values_wm, self.obs_sequences, self._num_observations_tokens = None, None, None

        self.temperature = 1.0

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        obs_sequences = self.module.encode_observations(self.tokenizer.preprocess_input(observations))
        obs_sequences = rearrange(obs_sequences, "b c h w -> b (h w) c")
        num_observations_tokens = obs_sequences.size(1)
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens

        _ = self.refresh_keys_values_with_initial_obs_tokens(obs_sequences)
        self.obs_sequences = obs_sequences

        return self.decode_obs_tokens()

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(self, obs_sequences):
        n, num_observations_tokens = obs_sequences.shape[:2]
        assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n,
                                                                                      max_tokens=self.world_model.config.max_tokens)
        outputs_wm = self.module.wm_forward(obs_sequences, past_keys_values=self.keys_values_wm)
        return outputs_wm

    def embed_action(self, action, action_continuous=None):
        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        token = token.reshape(-1, 1).to(self.device)  # (B, 1)
        if action_continuous is not None:
            action_continuous = action_continuous.clone().detach() if isinstance(action_continuous,
                                                                                 torch.Tensor) else torch.tensor(
                action_continuous,
                dtype=torch.float)
            action_continuous = action_continuous.reshape(-1, 1, self.world_model.act_continuous_size).to(
                self.device)  # (B, 1, #act_continuous)
        return self.world_model.embedder(token, 1, self.keys_values_wm.size,
                                         continuous=action_continuous)

    @torch.no_grad()
    def step(self, action: Union[int, torch.LongTensor],
             continuous=None):
        """

        :param action:
        :param continuous:
        :return:
        """
        max_context = self.world_model.config.max_tokens
        assert self.keys_values_wm is not None and self.num_observations_tokens is not None
        num_heads = self.world_model.config.n_gen_heads
        assert self.num_observations_tokens % num_heads == 0
        num_passes = self.num_observations_tokens // num_heads

        output_sequence = []

        if self.keys_values_wm.size + num_passes > max_context:
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_sequences)

        seq = self.embed_action(action, continuous)
        for k in range(num_passes):  # assumption that there is only one action token.
            seq = self.module.wm_forward(seq, past_keys_values=self.keys_values_wm)[0]
            output_sequence.append(seq)

        # add last tokens to context
        _, logits_rewards, logits_ends = self.module.wm_forward(seq, past_keys_values=self.keys_values_wm)
        reward = Categorical(logits=logits_rewards / self.temperature).sample()
        reward = self.world_model.decode_rewards(reward).float().cpu().numpy().reshape(-1)  # (B,)
        done = Categorical(logits=logits_ends / self.temperature).sample().cpu().numpy().astype(
            bool).reshape(-1)  # (B,)

        self.obs_sequences = torch.cat(output_sequence, dim=1)  # (B, K, E)
        obs = self.decode_obs_tokens()
        return obs, reward, done, None

    @torch.no_grad()
    def decode_obs_tokens(self):
        z = rearrange(self.obs_sequences, 'b (h w) e -> b e h w', h=int(math.sqrt(self.num_observations_tokens)))
        rec = self.module.decode_observations(z)  # (B, C, H, W)
        return torch.clamp(self.tokenizer.postprocess_output(rec), 0, 1)


@torch.no_grad()
def explore_world_model(checkpoint_path=None):
    wm_checkpoint_path = checkpoint_path or Path(r"")

    with hydra.initialize(config_path="../../config"):
        cfg = hydra.compose(config_name="joint")

    joint_wm = JointWM(cfg).to(device).eval()
    state_dict = torch.load(wm_checkpoint_path, map_location=device)

    joint_wm.load_state_dict(state_dict)

    dataset: EpisodesDataset = instantiate(cfg.datasets.train)
    dataset.max_num_episodes = int(1E9)
    dataset.load_disk_checkpoint(Path(r"input_artifacts\dataset"))

    wm_env = ContWMEnv(joint_wm)
    action_tokenizer = ActionTokenizer(move_shot_anchors=cfg.env.train.move_shot_anchors)

    def env_reset(ep_id=None, frame_id=None):
        ep_id = ep_id or random.randint(dataset.num_seen_episodes - len(dataset), dataset.num_seen_episodes - 1)
        episode = dataset.get_episode(ep_id)
        frame_id = frame_id or random.randint(0, len(episode) - 1)
        observation = episode.observations[[frame_id]]
        observation = observation / 255.0
        observation = torch.nn.functional.interpolate(observation, (dataset.resolution, dataset.resolution),
                                                      mode='bilinear')

        return wm_env.reset_from_initial_observations(observation.to(device))

    def env_step(make_move, make_shot, super_ability, use_gadget,
                 move_anchor, shot_anchor, move_shift=0, shot_shift=0, shot_strength=-10):
        token = action_tokenizer.create_action_token(
            make_move, make_shot, super_ability, use_gadget, move_anchor, shot_anchor
        )
        out = wm_env.step(token, continuous=[move_shift, shot_shift, shot_strength])
        return out

    ###
    # UI
    ###

    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
    import numpy as np

    IMG_SIZE = 368

    class Counter:
        pass

    counter = Counter()
    counter.value = 0

    def my_grid_func(obj, row, col, **kwargs):
        obj.grid_configure(row=row, column=col, **kwargs)

    tk.Grid.grid = my_grid_func

    window = tk.Tk()
    frame_master = tk.Frame(window, name="frame_master")
    frame_master.grid(0, 0)

    left_frame = tk.Frame(frame_master, name="left_frame")
    left_frame.grid(0, 0)

    placeholder_image = ImageTk.PhotoImage(Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255)))
    image_label = ttk.Label(left_frame, image=placeholder_image)
    image_label.image = placeholder_image
    image_label.grid(0, 0, pady=5, padx=10, sticky="n")

    def update_image(img):
        if isinstance(img, torch.Tensor):
            img = img.cpu().squeeze().permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
        p_image = ImageTk.PhotoImage(Image.fromarray(img, mode="RGB").resize((IMG_SIZE, IMG_SIZE)))
        image_label.config(image=p_image)
        image_label.image = p_image

    def after_step(*args):
        obs, reward, done = args[:3]
        update_image(obs)
        counter.value += 1
        info_var.set(f"Reward: {reward.item()}. Done: {done.item()}. Step: {counter.value}")

    def after_reset(*args):
        counter.value = 0
        update_image(*args)

    info_frame = tk.Frame(left_frame, name="info_frame")
    info_frame.grid(1, 0)
    info_var = tk.StringVar()
    info_label = tk.Label(info_frame, textvariable=info_var)
    info_label.grid(0, 0)

    controls_frame = tk.Frame(left_frame, name="controls_frame")
    controls_frame.grid(2, 0)
    tk.Label(controls_frame, text="Temperature ").grid(0, 0)
    temperature_var = tk.StringVar(value="1.0")
    ttk.Entry(controls_frame, textvariable=temperature_var).grid(0, 1)
    reset_button = ttk.Button(controls_frame, text="Reset", command=lambda: after_reset(*env_reset()))
    reset_button.grid(1, 0)
    step_button = ttk.Button(controls_frame, text="Step", command=lambda: after_step(*env_step(**gather_actions())))
    step_button.grid(1, 1)

    # action input
    right_frame = tk.Frame(window, name="right_frame")
    right_frame.grid(0, 1)
    action_input_frame = tk.Frame(right_frame, name="action_input_frame")
    action_input_frame.grid(0, 0)

    checkboxes = {}
    for i, name in enumerate(["make_move", "make_shot", "super_ability", "use_gadget"]):
        ch_var = tk.BooleanVar()
        ttk.Checkbutton(action_input_frame, text=name, variable=ch_var).grid(i, 0)
        checkboxes[name] = ch_var

    radiobuttons = {}
    for i, name in enumerate(["move_anchor", "shot_anchor"]):
        r_frame = tk.Frame(action_input_frame)
        r_frame.grid(len(checkboxes) + i, 0)
        tk.Label(r_frame, text=name).grid(0, 0, columnspan=4)
        r_var = tk.IntVar()
        for v in range(4):
            ttk.Radiobutton(r_frame, text=str(v), variable=r_var, value=v).grid(1, v)
        radiobuttons[name] = r_var

    def gather_actions():
        ch_values = {k: v.get() for k, v in checkboxes.items()}
        r_values = {k: v.get() for k, v in radiobuttons.items()}
        ch_values.update(r_values)
        wm_env.temperature = float(temperature_var.get())
        return ch_values

    reset_button.invoke()
    window.bind("<Return>", lambda x: step_button.invoke())
    window.bind("<Escape>", lambda x: reset_button.invoke())
    window.mainloop()


if __name__ == "__main__":
    main()
    explore_world_model(
        r"C:\Users\Michael\PycharmProjects\Brawl_iris\outputs\norm\2024-06-22_14-47-24\checkpoints\last.pt")
