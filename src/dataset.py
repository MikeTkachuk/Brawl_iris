from collections import deque
import math
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

import psutil
import torch
from einops import rearrange

from src.episode import Episode

Batch = Dict[str, torch.Tensor]


class EpisodesDataset:
    def __init__(self,
                 max_num_episodes: Optional[int] = None,
                 name: Optional[str] = None,
                 resolution: Optional[int] = None) -> None:
        self.max_num_episodes = max_num_episodes
        self.resolution = resolution
        self.name = name if name is not None else 'dataset'
        self.num_seen_episodes = 0
        self.episodes = deque()
        self.episode_id_to_queue_idx = dict()
        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()
        self.disk_episodes = deque()

    def __len__(self) -> int:
        return len(self.disk_episodes)

    def clear(self) -> None:
        self.episodes = deque()
        self.episode_id_to_queue_idx = dict()

    def add_episode(self, episode: Episode) -> int:
        print('dataset.Dataset.add_episode: ADD DATASET LOGGING')
        print(len(self.episodes), self.max_num_episodes)
        if self.max_num_episodes is not None and len(self.disk_episodes) == self.max_num_episodes:
            self._popleft()
        episode_id = self._append_new_episode(episode)
        print('dataset.Dataset.add_episode: AFTER ADD DATASET LOGGING')
        print(self.newly_deleted_episodes)
        return episode_id

    def get_episode(self, episode_id: int) -> Episode:
        assert episode_id in self.disk_episodes
        assert episode_id in self.episode_id_to_queue_idx, f"Episode {episode_id} is not loaded"
        queue_idx = self.episode_id_to_queue_idx[episode_id]
        return self.episodes[queue_idx]

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

    def sample_batch(self, batch_num_samples: int, sequence_length: int, weights: Optional[Tuple[float]] = None,
                     sample_from_start: bool = True) -> Batch:
        return self._collate_episodes_segments(
            self._sample_episodes_segments(batch_num_samples, sequence_length, weights, sample_from_start))

    def _sample_episodes_segments(self, batch_num_samples: int, sequence_length: int, weights: Optional[Tuple[float]],
                                  sample_from_start: bool) -> List[Episode]:
        num_episodes = len(self.episodes)
        num_weights = len(weights) if weights is not None else 0

        if num_weights < num_episodes:
            weights = [1] * num_episodes
        else:
            assert all([0 <= x <= 1 for x in weights]) and sum(weights) == 1
            sizes = [num_episodes // num_weights + (num_episodes % num_weights) * (i == num_weights - 1) for i in
                     range(num_weights)]
            weights = [w / s for (w, s) in zip(weights, sizes) for _ in range(s)]

        sampled_episodes = random.choices(self.episodes, k=batch_num_samples, weights=weights)

        sampled_episodes_segments = []
        for sampled_episode in sampled_episodes:
            if sample_from_start:
                start = random.randint(0, len(sampled_episode) - 1)
                stop = start + sequence_length
            else:
                stop = random.randint(1, len(sampled_episode))
                start = stop - sequence_length
            sampled_episodes_segments.append(sampled_episode.segment(start, stop, should_pad=True))
            assert len(sampled_episodes_segments[-1]) == sequence_length
        return sampled_episodes_segments

    def _collate_episodes_segments(self, episodes_segments: List[Episode]) -> Batch:
        episodes_segments = [e_s.__dict__ for e_s in episodes_segments]
        batch = {}
        for k in episodes_segments[0]:
            batch[k] = torch.stack([e_s[k] for e_s in episodes_segments])
        batch['observations'] = batch['observations'].float() / 255.0  # int8 to float and scale
        to_resize = rearrange(batch['observations'], 'b t ... -> (b t) ...')
        resized = torch.nn.functional.interpolate(to_resize, (self.resolution, self.resolution), mode='bilinear')
        batch['observations'] = rearrange(resized, '(b t) ... -> b t ...', b=batch['observations'].shape[0])
        return batch

    def traverse(self, batch_num_samples: int, chunk_size: int):
        for episode in self.episodes:
            chunks = [episode.segment(start=i * chunk_size, stop=(i + 1) * chunk_size, should_pad=True) for i in
                      range(math.ceil(len(episode) / chunk_size))]
            batches = [chunks[i * batch_num_samples: (i + 1) * batch_num_samples] for i in
                       range(math.ceil(len(chunks) / batch_num_samples))]
            for b in batches:
                yield self._collate_episodes_segments(b)

    def update_disk_checkpoint(self, directory: Path, flush=True) -> None:
        assert directory.is_dir()
        print(f'dataset.Dataset.update_disk_checkpoint: map: {self.episode_id_to_queue_idx}'
              f' n_mod: {self.newly_modified_episodes} n_del: {self.newly_deleted_episodes}')
        num_flushed = len(self.newly_modified_episodes)
        for episode_id in self.newly_modified_episodes:
            episode = self.get_episode(episode_id)
            episode_path = directory / f'{episode_id}.pt'
            episode.save(episode_path)
        for episode_id in self.newly_deleted_episodes:
            episode_path = directory / f'{episode_id}.pt'
            episode_path.unlink()
        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()

        # flush episodes to disk
        if flush:
            self.clear()
            print(f'dataset.Dataset.update_disk_checkpoint: flushed {num_flushed} episodes to disk')

    def load_disk_checkpoint(self, directory: Path, load_episodes=True) -> None:
        assert directory.is_dir()
        episode_ids = sorted([int(p.stem) for p in directory.iterdir()])

        if not len(episode_ids):
            self.disk_episodes = deque()
            self.num_seen_episodes = 0
            return

        self.disk_episodes = deque(episode_ids)
        self.num_seen_episodes = episode_ids[-1] + 1
        if load_episodes:
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
