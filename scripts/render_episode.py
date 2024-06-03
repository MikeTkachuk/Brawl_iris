from argparse import ArgumentParser
import time
from pathlib import Path

import numpy as np
import torch
import cv2 as cv

import sys

sys.path.append(r"C:\Users\Michael\PycharmProjects\Brawl_iris")
from src.models.tokenizer import Tokenizer
from src.models.actor_critic import ActorCritic


def make_video(fname, fps, frames):
    assert frames.ndim == 4  # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3

    video = cv.VideoWriter(str(fname), cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        video.write(frame[:, :, ::-1])
    video.release()


def show_video(frames, fps=1, annotate=False, captions=None, tokenize=True):
    if captions is None:
        captions = [""] * len(frames)
    assert frames.ndim == 4  # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3
    if annotate:
        actor = load_critic(
            r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\actor_mlp_head\2024-03-31_22-44-53\checkpoints\last.pt")
        actor.reset(1)
    if tokenize:
        from hydra.utils import instantiate
        import hydra
        with hydra.initialize(config_path=r"../config"):
            cfg = hydra.compose(config_name="trainer")
        tokenizer = instantiate(cfg.tokenizer)
        tokenizer.load_state_dict(
            torch.load(r"C:\Users\Michael\PycharmProjects\Brawl_iris\input_artifacts\tokenizer.pt"))
        tokenizer.to("cuda").eval()

    try:
        import win32api as wapi
        def break_func():
            if wapi.GetAsyncKeyState(ord("Q")):
                exit()
    except ImportError:
        break_func = lambda: ()

    for frame, caption in zip(frames, captions):
        start = time.time()
        if tokenize:
            frame = tokenize_frame(tokenizer, frame)
        if annotate:
            annotated = annotate_frame_value(actor, frame[..., ::-1].copy())
        else:
            annotated = annotate_frame(frame[..., ::-1], text=caption)
        cv.imshow("Episode", annotated)
        elapsed = time.time() - start
        time.sleep(max(0.0, 1 / fps - elapsed))
        break_func()
        if cv.waitKey(25):
            pass
    cv.destroyAllWindows()


def tokenize_frame(tokenizer: Tokenizer, frame):
    inp = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).to("cuda")[None, ...] / 255.0
    inp = torch.nn.functional.interpolate(inp, (tokenizer.encoder.config.resolution,
                                                tokenizer.encoder.config.resolution),
                                    mode='bilinear')
    tokenized = tokenizer.encode_decode(inp, True, True)
    return tokenized[0].permute(1, 2, 0).cpu().clamp(0, 1).mul(255.0).byte().numpy()


def load_critic(path):
    actor_weights = torch.load(path)
    field_name = "actor_critic."
    actor_weights = {k[len(field_name):]: v for k, v in actor_weights.items() if field_name in k}
    actor = ActorCritic(None, 3)
    actor.load_state_dict(actor_weights)
    return actor.to("cuda")


@torch.no_grad()
def annotate_frame_value(actor: ActorCritic, frame):
    inp = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).to("cuda")[None, ...] / 255.0
    value = actor.forward(inp).means_values.item()
    frame = cv.resize(frame, (512, 512))
    frame = cv.putText(frame, f"{value:.2f}", (400, 50), cv.FONT_HERSHEY_SIMPLEX,
                       1, (255, 255, 255))
    return frame


def annotate_frame(frame, text=""):
    frame = cv.resize(frame, (400, 400))
    frame = cv.putText(frame, text, (50, 350), cv.FONT_HERSHEY_SIMPLEX,
                       1, (255, 255, 255))
    return frame


def save_episode_as_image_folder(path, out_dir):
    ep = torch.load(path)
    ep_folder = Path(out_dir) / Path(path).stem
    ep_folder.mkdir(parents=True, exist_ok=True)
    for i in range(len(ep["observations"])):
        cv.imwrite(str(ep_folder / f"{i}.png"), ep["observations"][i].permute(1, 2, 0).numpy()[..., ::-1])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--fps", type=float, default=2)
    args = parser.parse_args()

    ep = torch.load(args.path)
    move_directions = ["right", "up", "left", "down"]
    show_video(ep["observations"].permute(0, 2, 3, 1).numpy(),
               captions=[move_directions[a[-2]] if a[0] else "No action" for a in ep["actions"]],
               fps=args.fps)
