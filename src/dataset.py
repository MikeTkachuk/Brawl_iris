import os
from collections import deque
import math
from multiprocessing import Pool
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple
from functools import partial

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from einops import rearrange
from torchvision.transforms.functional import gaussian_blur

from tqdm import tqdm

from src.episode import Episode

Batch = Dict[str, torch.Tensor]


class _Dataset(Dataset):
    def __init__(self, episode_ids, episode_dir, resolution=None):
        super().__init__()
        self.episode_ids = episode_ids
        self.episode_dir = Path(episode_dir)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, idx):
        episode_id = self.episode_ids[idx]
        return Episode(**torch.load(self.episode_dir / f'{episode_id}.pt')), episode_id


def _tokenizer_preprocess_helper(ep_id, episode_dir, resolution, preprocessed_dir):
    episode = Episode(**torch.load(episode_dir / f'{ep_id}.pt'))
    observations = episode.observations / 255.0
    observations = torch.nn.functional.interpolate(observations, (resolution, resolution),
                                                   mode='bilinear')
    op_flow = torch.diff(observations, dim=0).abs()
    weights = gaussian_blur(op_flow, 5).max(dim=-3)[0]
    weights = weights + 1 - weights.mean()
    weights = weights ** 1.2
    sample_paths = []
    for i in range(1, len(episode)):
        sample_name = f"{ep_id}_{i}.pt"
        sample_path = preprocessed_dir / sample_name
        sample = {"frame": observations[[i]], "weights": weights[[i - 1]]}
        torch.save(sample, sample_path)
        sample_paths.append(sample_path)
    return sample_paths


class _TokenizerDataset(Dataset):
    def __init__(self, episode_ids, episode_dir, resolution):
        super().__init__()
        self.episode_ids = episode_ids
        self.episode_dir = Path(episode_dir)
        self.resolution = resolution
        self._preprocess()

    def _preprocess(self):
        self.preprocessed_dir = self.episode_dir / "preprocessed"
        self.preprocessed_dir.mkdir()
        with Pool(8) as workers:
            self.sample_paths = sum(
                tqdm(workers.imap(
                    partial(_tokenizer_preprocess_helper,
                            episode_dir=self.episode_dir,
                            resolution=self.resolution,
                            preprocessed_dir=self.preprocessed_dir),
                    self.episode_ids,
                    chunksize=8
                ), desc="Precomputing data: ", total=len(self.episode_ids)),
                [])

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample = torch.load(self.sample_paths[idx])
        return sample["frame"], sample["weights"]


def _collate_fn(samples, resolution=None, sample_segments=False, segment_len=None, end_proba=0.1, tokenizer=False):
    if tokenizer:
        segments, diffs = list(zip(*samples))
        collated = torch.stack(segments)
        diff = torch.stack(diffs)
        return {"observations": collated}, diff

    episodes, ids = list(zip(*samples))
    max_len = max(len(ep) for ep in episodes)
    sampled_episodes_segments = []
    for sampled_episode in episodes:
        if not sample_segments:
            start = len(sampled_episode) - max_len
            stop = len(sampled_episode)
        else:
            assert segment_len is not None
            if random.random() > end_proba:
                stop = random.randint(min(segment_len, len(sampled_episode)), len(sampled_episode))
            else:
                stop = len(sampled_episode)
            start = stop - segment_len
        sampled_episodes_segments.append(sampled_episode.segment(start, stop, should_pad=True))

    collated = EpisodesDataset.collate_episodes_segments(sampled_episodes_segments, resolution=resolution)
    return collated, ids


class EpisodesDataset:
    def __init__(self,
                 max_num_episodes: Optional[int] = None,
                 name: Optional[str] = None,
                 resolution: Optional[int] = None,
                 lazy: bool = False
                 ) -> None:
        self.max_num_episodes = max_num_episodes
        self.resolution = resolution
        self.name = name if name is not None else 'dataset'
        self.num_seen_episodes = 0
        self.episodes = deque()
        self.episode_id_to_queue_idx = dict()
        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()
        self.disk_episodes = deque()
        self.lazy = lazy
        self.episode_weights = {}

        self._dir = None

    def __len__(self) -> int:
        return len(self.disk_episodes)

    def clear(self) -> None:
        self.episodes = deque()
        self.episode_id_to_queue_idx = dict()

    def add_episode(self, episode: Episode, weight=0) -> int:
        print('dataset.Dataset.add_episode: ADD DATASET LOGGING')
        print(len(self.episodes), self.max_num_episodes)
        if self.max_num_episodes is not None and len(self.disk_episodes) == self.max_num_episodes:
            self._popleft()
        episode_id = self._append_new_episode(episode)
        self.episode_weights[episode_id] = weight
        print('dataset.Dataset.add_episode: AFTER ADD DATASET LOGGING')
        print(self.newly_deleted_episodes)
        return episode_id

    def get_episode(self, episode_id: int) -> Episode:
        assert episode_id in self.disk_episodes
        if episode_id in self.episode_id_to_queue_idx:
            queue_idx = self.episode_id_to_queue_idx[episode_id]
            return self.episodes[queue_idx]
        if self.lazy:
            assert self._dir is not None, "Dataset was not loaded properly"
            episode = Episode(**torch.load(self._dir / f'{episode_id}.pt'))
            return episode
        raise Exception(f"Episode {episode_id} is not loaded")

    def _popleft(self) -> int:
        id_to_delete = self.disk_episodes.popleft()
        self.newly_deleted_episodes.add(id_to_delete)
        self.newly_modified_episodes.discard(id_to_delete)  # in case an episode is created and deleted in one epoch
        if id_to_delete in self.episode_id_to_queue_idx:
            self.episode_id_to_queue_idx = {k: v - 1 for k, v in self.episode_id_to_queue_idx.items() if v > 0}
            self.episodes.popleft()
        return id_to_delete

    def _append_new_episode(self, episode):
        episode_id = self.num_seen_episodes
        self.episode_id_to_queue_idx[episode_id] = len(self.episodes)
        self.episodes.append(episode)
        self.disk_episodes.append(episode_id)
        self.num_seen_episodes += 1
        self.newly_modified_episodes.add(episode_id)
        return episode_id

    def torch_dataloader(self, batch_size,
                         epochs=100_000,
                         random_sampling=True,
                         parallelize=True,
                         num_workers=2,
                         prefetch_factor=1,
                         sample_segments=False,
                         segment_len=None,
                         pin_memory=False,
                         tokenizer=False):
        """Static torch version of dataset. Captures disk episodes
        :param tokenizer: if true, segment_len will be set to 2, optical flow mask will be produced
        """

        class _BatchSampler(Sampler):
            def __init__(self, data, batch_size, random_sampling=True, shuffle=True):
                self.data = data
                self.batch_size = batch_size
                self.random_sampling = random_sampling
                self.shuffle = shuffle

            def __len__(self):
                return epochs if self.random_sampling else int(np.ceil(len(self.data) / self.batch_size))

            def __iter__(self):
                pool = np.arange(len(self.data))
                if self.random_sampling:
                    for _ in range(len(self)):
                        yield np.random.choice(pool, size=(self.batch_size,), replace=False).tolist()
                else:
                    if self.shuffle:
                        pool = np.random.permutation(pool)
                    for i in range(len(self)):
                        yield pool[i * self.batch_size: (i + 1) * self.batch_size]

        dataset = (_TokenizerDataset if tokenizer else _Dataset)(self.disk_episodes, self._dir, self.resolution)
        return DataLoader(
            dataset,
            batch_sampler=_BatchSampler(dataset, batch_size, random_sampling=random_sampling, shuffle=True),
            collate_fn=partial(_collate_fn, resolution=self.resolution,
                               sample_segments=sample_segments, segment_len=segment_len, tokenizer=tokenizer),
            num_workers=num_workers if parallelize else 0,
            persistent_workers=True if parallelize else False,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if parallelize else 2,  # should be 2 if not used (weird pytorch check)
        )

    def sample_replay(self, batch_num_samples: int = None, weights: Optional[Tuple[float]] = None, samples=None):
        if samples is None:
            episode_pool = self.episodes if not self.lazy else self.disk_episodes
            num_episodes = len(episode_pool)
            num_weights = len(weights) if weights is not None else 0

            assert all([0 <= x <= 1 for x in weights]) and sum(weights) == 1
            sizes = [num_episodes // num_weights + (num_episodes % num_weights) * (i == num_weights - 1) for i in
                     range(num_weights)]
            weights = [w / s for (w, s) in zip(weights, sizes) for _ in range(s)]
            sampled_episodes = random.choices(episode_pool, k=batch_num_samples, weights=weights)
            if self.lazy:
                sampled_episodes = [self.get_episode(ep) for ep in sampled_episodes]
        else:
            sampled_episodes = [self.get_episode(i) for i in samples]

        max_len = max(len(ep) for ep in sampled_episodes)
        sampled_episodes_segments = []

        for sampled_episode in sampled_episodes:
            start = len(sampled_episode) - max_len
            stop = len(sampled_episode)
            sampled_episodes_segments.append(sampled_episode.segment(start, stop, should_pad=True))

        return self.collate_episodes_segments(sampled_episodes_segments, self.resolution)

    def sample_batch(self, batch_num_samples: int, sequence_length: int, weights: Optional[Tuple[float]] = None,
                     sample_from_start: bool = True) -> Batch:
        return self.collate_episodes_segments(
            self._sample_episodes_segments(batch_num_samples, sequence_length, weights, sample_from_start),
            self.resolution)

    def _sample_episodes_segments(self, batch_num_samples: int, sequence_length: int, weights: Optional[Tuple[float]],
                                  sample_from_start: bool) -> List[Episode]:
        episode_pool = self.episodes if not self.lazy else self.disk_episodes
        num_episodes = len(episode_pool)
        num_weights = len(weights) if weights is not None else 0

        if num_weights < num_episodes:
            weights = [1] * num_episodes
        else:
            assert all([0 <= x <= 1 for x in weights]) and sum(weights) == 1
            sizes = [num_episodes // num_weights + (num_episodes % num_weights) * (i == num_weights - 1) for i in
                     range(num_weights)]
            weights = [w / s for (w, s) in zip(weights, sizes) for _ in range(s)]

        sampled_episodes = random.choices(episode_pool, k=batch_num_samples, weights=weights)
        if self.lazy:
            sampled_episodes = [self.get_episode(ep) for ep in sampled_episodes]
        sampled_episodes_segments = []
        for sampled_episode in sampled_episodes:
            if sample_from_start:
                start = random.randint(0, len(sampled_episode) - 1)
                stop = start + sequence_length
            else:
                stop = random.randint(min(39, len(sampled_episode)), len(sampled_episode))  # 39 = total - burn in - 1
                start = stop - sequence_length
            sampled_episodes_segments.append(sampled_episode.segment(start, stop, should_pad=True))
            assert len(sampled_episodes_segments[-1]) == sequence_length
        return sampled_episodes_segments

    @staticmethod
    def collate_episodes_segments(episodes_segments: List[Episode], resolution=None) -> Batch:
        episodes_segments = [e_s.__dict__ for e_s in episodes_segments]
        batch = {}
        for k in episodes_segments[0]:
            if isinstance(episodes_segments[0][k], torch.Tensor):
                batch[k] = torch.stack([e_s[k] for e_s in episodes_segments])
        if resolution is not None and not torch.all(batch["observations"].shape[-2:] == resolution):
            to_resize = rearrange(batch['observations'], 'b t ... -> (b t) ...')
            resized = torch.nn.functional.interpolate(to_resize, (resolution, resolution),
                                                      mode='nearest')  # torch 2.1 supports uint8 bilinear
            batch['observations'] = rearrange(resized, '(b t) ... -> b t ...', b=batch['observations'].shape[0])
        batch['observations'] = batch['observations'].float() / 255.0  # int8 to float and scale
        return batch

    def traverse(self, batch_num_samples: int, chunk_size: int):
        episode_pool = self.episodes if not self.lazy else self.disk_episodes
        for episode in episode_pool:
            if self.lazy:
                episode = self.get_episode(episode)
            chunks = [episode.segment(start=i * chunk_size, stop=(i + 1) * chunk_size, should_pad=True) for i in
                      range(math.ceil(len(episode) / chunk_size))]
            batches = [chunks[i * batch_num_samples: (i + 1) * batch_num_samples] for i in
                       range(math.ceil(len(chunks) / batch_num_samples))]
            for b in batches:
                yield self.collate_episodes_segments(b, self.resolution)

    def update_disk_checkpoint(self, directory: Path, flush=True) -> None:
        assert directory.is_dir()
        self._dir = directory
        print(f'dataset.Dataset.update_disk_checkpoint: map: {self.episode_id_to_queue_idx}'
              f' n_mod: {self.newly_modified_episodes} n_del: {self.newly_deleted_episodes}')
        torch.save(self.episode_weights,
                   directory / "weights.pt")  # save before episodes to avoid key errors during concurrency
        num_flushed = len(self.newly_modified_episodes)
        for episode_id in self.newly_modified_episodes:
            episode = self.get_episode(episode_id)
            episode_path = directory / f'{episode_id}.pt'
            episode.save(episode_path)
        for episode_id in self.newly_deleted_episodes:
            episode_path = directory / f'{episode_id}.pt'
            episode_path.unlink()
            self.episode_weights.pop(episode_id)

        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()

        # flush episodes to disk
        if flush or self.lazy:
            self.clear()
            print(f'dataset.Dataset.update_disk_checkpoint: flushed {num_flushed} episodes to disk')

    def load_disk_checkpoint(self, directory: Path, load_episodes=True) -> None:
        assert directory.is_dir()
        assert not self.episodes or self.lazy, "Repetitive loading"
        episode_ids = sorted([int(p.stem) for p in directory.iterdir() if p.stem not in ["weights"]])

        if not len(episode_ids):
            self.disk_episodes = deque()
            self.num_seen_episodes = 0
            return

        self.disk_episodes = deque(episode_ids)
        self._dir = directory
        self.num_seen_episodes = episode_ids[-1] + 1
        try:
            self.episode_weights = torch.load(directory / "weights.pt")
        except Exception as e:
            print("While loading dataset could not load weights. Setting to zero")
            print("Error: ", e)
            self.episode_weights = {i: 0.0 for i in episode_ids}

        if load_episodes and not self.lazy:
            for episode_id in episode_ids:
                episode = Episode(**torch.load(directory / f'{episode_id}.pt'))
                self.episode_id_to_queue_idx[episode_id] = len(self.episodes)
                self.episodes.append(episode)


class EpisodesDatasetRamMonitoring(EpisodesDataset):
    """
    Prevent episode dataset from going out of RAM.
    Warning: % looks at system wide RAM usage while G looks only at process RAM usage.
    """

    def __init__(self, max_ram_usage: str, name: Optional[str] = None) -> None:
        super().__init__(max_num_episodes=None, name=name)
        self.max_ram_usage = max_ram_usage
        self.num_steps = 0
        self.max_num_steps = None

        max_ram_usage = str(max_ram_usage)
        if max_ram_usage.endswith('%'):
            m = int(max_ram_usage.split('%')[0])
            assert 0 < m < 100
            self.check_ram_usage = lambda: psutil.virtual_memory().percent > m
        else:
            assert max_ram_usage.endswith('G')
            m = float(max_ram_usage.split('G')[0])
            self.check_ram_usage = lambda: psutil.Process().memory_info()[0] / 2 ** 30 > m

    def clear(self) -> None:
        super().clear()
        self.num_steps = 0

    def add_episode(self, episode: Episode) -> int:
        if self.max_num_steps is None and self.check_ram_usage():
            self.max_num_steps = self.num_steps
        self.num_steps += len(episode)
        while (self.max_num_steps is not None) and (self.num_steps > self.max_num_steps):
            self._popleft()
        episode_id = self._append_new_episode(episode)
        return episode_id

    def _popleft(self) -> Episode:
        episode = super()._popleft()
        self.num_steps -= len(episode)
        return episode
