from argparse import ArgumentParser
import time
from pathlib import Path

import numpy as np
import torch
import cv2 as cv

import sys

sys.path.append(r"C:\Users\Michael\PycharmProjects\Brawl_iris")
from src.models.actor_critic import ActorCritic


def make_video(fname, fps, frames):
    assert frames.ndim == 4  # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3

    video = cv.VideoWriter(str(fname), cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        video.write(frame[:, :, ::-1])
    video.release()


def show_video(frames, fps=1, annotate=False):
    assert frames.ndim == 4  # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3
    if annotate:
        actor = load_critic(
            r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\actor_mlp_head\2024-03-31_22-44-53\checkpoints\last.pt")
        actor.reset(1)

    for frame in frames:
        start = time.time()
        if annotate:
            annotated = annotate_frame(actor, frame[..., ::-1].copy())
        else:
            annotated = frame[..., ::-1]
        cv.imshow("Episode", annotated)
        elapsed = time.time() - start
        time.sleep(max(0.0, 1 / fps - elapsed))
        if cv.waitKey(25):
            pass
    cv.destroyAllWindows()


def load_critic(path):
    actor_weights = torch.load(path)
    field_name = "actor_critic."
    actor_weights = {k[len(field_name):]: v for k, v in actor_weights.items() if field_name in k}
    actor = ActorCritic(None, 3)
    actor.load_state_dict(actor_weights)
    return actor.to("cuda")


@torch.no_grad()
def annotate_frame(actor: ActorCritic, frame):
    inp = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).to("cuda")[None, ...] / 255.0
    value = actor.forward(inp).means_values.item()
    frame = cv.resize(frame, (512, 512))
    frame = cv.putText(frame, f"{value:.2f}", (400, 50), cv.FONT_HERSHEY_SIMPLEX,
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
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    ep = torch.load(args.path)
    show_video(ep["observations"].permute(0, 2, 3, 1).numpy())
