import math
import random
import shutil
import sys
from pathlib import Path
import time
import os

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
from src.utils import set_seed, adaptive_gradient_clipping, ActionTokenizer
from src.models.tokenizer.tokenizer import Tokenizer
from src.models.world_model import WorldModel

device = "cuda"


@torch.no_grad()
def generate_token_dataset():
    with hydra.initialize(config_path="../../config"):
        cfg = hydra.compose(config_name="trainer")
    action_tokenizer = ActionTokenizer(move_shot_anchors=cfg.env.train.move_shot_anchors)
    tokenizer: Tokenizer = instantiate(cfg.tokenizer)
    tokenizer.load_state_dict(
        torch.load(r"input_artifacts\tokenizer.pt")
    )
    tokenizer.to(device).eval()

    dataset: EpisodesDataset = instantiate(cfg.datasets.train)
    dataset.load_disk_checkpoint(Path(r"input_artifacts\dataset"))
    token_dataset = {}
    for ep_id in tqdm(dataset.disk_episodes, desc="Episodes: "):
        episode = dataset.get_episode(ep_id)
        observations = episode.observations / 255.0
        observations = torch.nn.functional.interpolate(observations, (dataset.resolution, dataset.resolution),
                                                       mode='bilinear')
        all_tokens = []
        b_size = 64
        for b_id in range(math.ceil(len(episode) / b_size)):
            tokens = tokenizer.encode(observations[None, b_id * b_size: (b_id + 1) * b_size].to(device),
                                      should_preprocess=True).tokens
            all_tokens.append(tokens)
        tokens = torch.cat(all_tokens, dim=1)[0]
        action_tokens = [action_tokenizer.create_action_token(*a.cpu().numpy()) for a in
                         episode.actions]
        token_dataset[ep_id] = {
            "tokens": tokens,
            "actions": torch.LongTensor(action_tokens),
            "actions_continuous": episode.actions_continuous,
            "rewards": episode.rewards,
            "ends": episode.ends
        }
    torch.save(token_dataset, r"input_artifacts/token_dataset.pt")


def get_dataloader(dataset: dict, batch_size, segment_len, steps_per_epoch=None):
    episode_pool = list(dataset.keys())

    def _segment(episode_dict: dict, start, stop):
        episode_len = len(episode_dict["ends"])
        assert 0 <= start <= episode_len
        pad_right = max(0, stop - episode_len)
        true_slice = slice(start, stop - pad_right)
        out = {k: torch.nn.functional.pad(v[true_slice], [0] * (2 * v.ndim - 1) + [pad_right]) for k, v in
               episode_dict.items()}
        mask = torch.ones((stop - start,), dtype=torch.bool)
        mask[stop - start - pad_right:] = False
        out["mask_padding"] = mask
        return out

    def _sample():
        sampled_episodes = random.choices(episode_pool, k=batch_size)
        segments = []
        for ep_id in sampled_episodes:
            sampled_episode: dict = dataset[ep_id]
            episode_len = len(sampled_episode["ends"])
            start = random.randint(0, max(0, episode_len - segment_len))
            stop = start + segment_len
            segments.append(_segment(sampled_episode, start, stop))
        out = {k: torch.stack([s[k] for s in segments]) for k in segments[0]}
        out["actions"] = out["actions"][..., None]
        return out

    def dataloader():
        step = 0
        while steps_per_epoch is None or step < steps_per_epoch:
            yield _sample()
            step += 1

    return dataloader()


def custom_setup(cfg):
    torch.backends.cudnn.benchmark = True
    set_seed(cfg.common.seed)
    if sys.gettrace() is not None:  # if debugging
        cfg.wandb.mode = "offline"
        cfg.training.world_model.batch_num_samples = 2

    cfg.wandb.tags = list(set(cfg.wandb.tags or [] + ["world_model"]))
    wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
               reinit=True,
               resume=True,
               **cfg.wandb)


@hydra.main(config_path="../../config", config_name="trainer")
def main(cfg):
    try:
        print(f"PID: {os.getpid()}")
        Path("checkpoints/dataset").mkdir(parents=True, exist_ok=True)

        tokenizer: Tokenizer = instantiate(cfg.tokenizer)
        action_tokenizer = ActionTokenizer(move_shot_anchors=cfg.env.train.move_shot_anchors)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=action_tokenizer.n_actions,
                                 act_continuous_size=3,
                                 config=instantiate(cfg.world_model)).to(device).train()
        optimizer = torch.optim.Adam(world_model.parameters(),
                                     lr=cfg.training.learning_rate)

        # load checkpoint
        # world_model.load_state_dict(torch.load(r"C:\Users\Michael\PycharmProjects\Brawl_iris\outputs\world_model\2024-05-04_21-35-54\checkpoints\last.pt", map_location=device))
        # optimizer.load_state_dict(torch.load(r"C:\Users\Michael\PycharmProjects\Brawl_iris\outputs\world_model\2024-05-04_21-35-54\checkpoints\optimizer.pt", map_location=device))

        dataset = torch.load(Path(SOURCE_ROOT) / "input_artifacts/token_dataset.pt")
        custom_setup(cfg)
        train_keys, eval_keys = train_test_split(list(dataset.keys()), test_size=0.15)
        train_dataset = {k: dataset[k] for k in train_keys}
        dataloader = get_dataloader(train_dataset, batch_size=cfg.training.world_model.batch_num_samples,
                                    segment_len=cfg.common.sequence_length)
        eval_dataset = {k: dataset[k] for k in eval_keys}
        for n_step in tqdm(range(100_000), desc="Steps: "):
            batch = next(dataloader)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = world_model.compute_loss(batch, None)
            loss.loss_total.backward()
            if (n_step + 1) % cfg.training.world_model.grad_acc_steps == 0:
                adaptive_gradient_clipping(world_model.parameters(), lam=cfg.training.world_model.agc_lambda)
                optimizer.step()
                for p in world_model.parameters():
                    p.grad = None

                to_log = {}
                to_log.update(loss.intermediate_losses)
                to_log = {f"train/{k}": v for k, v in to_log.items()}
                if (n_step + 1) % (cfg.training.world_model.grad_acc_steps*4) == 0:
                    wandb.log(to_log)

            if n_step % (500 * cfg.training.world_model.grad_acc_steps) == 0:
                eval_dataloader = get_dataloader(eval_dataset, batch_size=cfg.training.world_model.batch_num_samples,
                                                 segment_len=cfg.common.sequence_length, steps_per_epoch=100)
                world_model.eval()
                eval_metrics = []
                with torch.no_grad():
                    for batch in eval_dataloader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        eval_metrics.append(world_model.compute_loss(batch, None).intermediate_losses)
                avg_eval_metrics = {f"eval/{k}": sum([m[k] for m in eval_metrics]) / len(eval_metrics) for k in eval_metrics[0]}
                wandb.log(avg_eval_metrics)
                world_model.train()

            if (n_step + 1) % (100 * cfg.training.world_model.grad_acc_steps) == 0:
                torch.save(world_model.state_dict(), "checkpoints/last.pt")
                torch.save(optimizer.state_dict(), "checkpoints/optimizer.pt")
            if (n_step + 1) % (1000 * cfg.training.world_model.grad_acc_steps) == 0:
                Path(f"checkpoints{n_step // 1000}").mkdir()
                torch.save(world_model.state_dict(), f"checkpoints{n_step // 1000}/last.pt")
                torch.save(optimizer.state_dict(), f"checkpoints{n_step // 1000}/optimizer.pt")

    finally:
        wandb.finish()


@hydra.main(config_path="../../config", config_name="trainer")
def main_ac_head(cfg):
    # todo: train separate attention head as an AC. Distill end/reward knowledge
    pass


@torch.no_grad()
def explore_world_model(checkpoint_path=None, max_context=None):
    from src.envs.world_model_env import WorldModelEnv
    wm_checkpoint_path = checkpoint_path or Path(
        r"C:\Users\Michael\PycharmProjects\Brawl_iris\outputs\world_model\2024-05-04_21-35-54\checkpoints\last.pt")

    with hydra.initialize(config_path="../../config"):
        cfg = hydra.compose(config_name="trainer")

    tokenizer: Tokenizer = instantiate(cfg.tokenizer)
    tokenizer.load_state_dict(
        torch.load(r"input_artifacts\tokenizer.pt")
    )
    tokenizer.to(device).eval()
    action_tokenizer = ActionTokenizer(move_shot_anchors=cfg.env.train.move_shot_anchors)
    world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=action_tokenizer.n_actions,
                             act_continuous_size=3,
                             config=instantiate(cfg.world_model)).to(device).eval()
    world_model.load_state_dict(torch.load(wm_checkpoint_path, map_location=device), strict=False)
    dataset: EpisodesDataset = instantiate(cfg.datasets.train)
    dataset.load_disk_checkpoint(Path(r"input_artifacts\dataset"))

    wm_env = WorldModelEnv(tokenizer, world_model, device)
    action_tokenizer = ActionTokenizer(move_shot_anchors=cfg.env.train.move_shot_anchors)

    def env_reset(ep_id=None, frame_id=None):
        ep_id = ep_id or random.randint(0, len(dataset))
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
        tok_b = wm_env.obs_tokens.reshape(12, 12)
        out = wm_env.step(token, continuous=[move_shift, shot_shift, shot_strength], context_shift=max_context or True)
        print(wm_env.obs_tokens.reshape(12, 12) - tok_b)
        return out

    ###
    # UI
    ###

    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
    import numpy as np

    IMG_SIZE = 350

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
        p_image = ImageTk.PhotoImage(Image.fromarray(img, mode="RGB"))
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
    # todo: add eval to tokenizer and wm (add more episodes)
    # todo: collect true random behaviour with fixed movement token (20% time standing)
    # todo: 0.5s frame too easy, wm collapses into copying previous frame. Train on 2 action gap (attention mask)
    # ep = torch.load(r"C:\Users\Michael\PycharmProjects\Brawl_iris\input_artifacts\dataset\43.pt")
    # observations = ep["observations"]
    # import matplotlib.pyplot as plt
    # for i in range(1, len(ep["ends"])):
    #     img = torch.cat([observations[i-1], observations[i]], dim=-1).permute(1,2,0).numpy()
    #     plt.imshow(img)
    #     plt.title(ep["actions"][i-1])
    #     plt.show()
    # main()
    explore_world_model(r"C:\Users\Michael\Downloads\last.pt",
                        max_context=None)
    # generate_token_dataset()
