from collections import namedtuple
from dataclasses import dataclass

import torch


@dataclass
class WordStats:
    distances: torch.Tensor
    counts: torch.Tensor

    def __post_init__(self):
        self.global_unused = torch.zeros_like(self.counts)

    def pop(self, idx):
        """
        Pop elements at given index or using mask
        :param idx: index or mask tensor. Pops entries at True locations
        :return: tuple of popped elements
        """
        out = (self.distances[idx], self.counts[idx])
        if isinstance(idx, int) or idx.ndim == 0:
            self.distances = torch.cat([self.distances[:idx], self.distances[idx + 1:]])
            self.counts = torch.cat([self.counts[:idx], self.counts[idx + 1:]])
            self.global_unused = torch.cat([self.global_unused[:idx], self.global_unused[idx + 1:]])
        else:
            self.counts = self.counts[~idx]
            self.distances = self.distances[~idx]
            self.global_unused = self.global_unused[~idx]

        return out

    def update(self, idx, distance):
        # TODO (optional) try median
        count = self.counts[idx]
        self.distances[idx] *= count / (1 + count)
        self.distances[idx] += distance / (1 + count)
        self.counts[idx] += 1

    def reset(self):
        self.global_unused[self.counts == 0] += 1
        self.distances = torch.zeros_like(self.distances)
        self.counts = torch.zeros_like(self.counts)

    def add(self, num=1):
        assert num > 0
        self.global_unused = torch.cat([self.global_unused, torch.zeros(num, dtype=self.global_unused.dtype)])
        self.distances = torch.cat([self.distances, torch.zeros(num, dtype=self.distances.dtype)])
        self.counts = torch.cat([self.counts, torch.zeros(num, dtype=self.counts.dtype)])


class DynamicVocab(torch.nn.Module):
    """
        After each epoch call step() to:
         - remove unused words for more than specified step() calls
         - join any two words if their distance and personal deviation is
         less than dst_tolerance / 3 (so that the resulting word deviated by at most dst_tolerance)
         - splits words in two if their deviation is greater than dst_tolerance.

    """

    def __init__(self,
                 init_vocab_size,
                 emb_size,
                 max_size=100_000,
                 dst_tolerance=0.01,
                 usage_patience=2,
                 max_add_per_step=-1):
        super(DynamicVocab, self).__init__()

        assert init_vocab_size < max_size
        self.table = torch.nn.Embedding(init_vocab_size, emb_size)
        self.max_size = max_size
        self.max_add_per_step = max_add_per_step

        ToleranceConfig = namedtuple('ToleranceConfig', ['distance', 'usage'])
        assert dst_tolerance > 0
        assert usage_patience > 0
        self.tolerance = ToleranceConfig(dst_tolerance, usage_patience)

        self._word_stats = WordStats(distances=torch.zeros(init_vocab_size, dtype=torch.float32),
                                     counts=torch.zeros(init_vocab_size, dtype=torch.int32))

    def __len__(self):
        return self.table.weight.shape[0]

    @property
    def weight(self):
        return self.table.weight

    def forward(self, *args, **kwargs):
        return self.table(*args, **kwargs)

    def map_closest(self, x):
        """

        :param x: of shape N, emb_dim
        :return: N tokens
        """
        dist_to_embeddings = torch.sum(x ** 2, dim=1, keepdim=True) + \
                             torch.sum(self.weight ** 2, dim=1) - 2 * torch.matmul(x, self.weight.t())
        distances, tokens = dist_to_embeddings.max(dim=-1)
        for token, distance in zip(tokens, distances):
            self._word_stats.update(token, distance)

    def reset(self):
        self._word_stats.reset()

    def pop(self, idx):
        self._word_stats.pop(idx)
        if isinstance(idx, int) or idx.ndim == 0:
            self.table.weight = torch.cat([self.table.weight[:idx], self.table.weight[idx + 1:]])
        else:
            self.table.weight = self.table.weight[~idx]

    def add(self, vec):
        vec = vec.reshape(-1, self.weight.shape[1])
        self.table.weight = torch.cat([self.table.weight, vec])
        self._word_stats.add(num=vec.shape[0])

    def step(self):
        # remove unused first
        remove_mask = self._word_stats.global_unused > self.tolerance.usage
        print(f'vocabulary.DynamicVocab.step: removing {remove_mask.nonzero()[:, 0]}')
        self.pop(remove_mask)

        # join similar
        deviation_mask = self._word_stats.distances < self.tolerance.distance / 3
        deviation_mask = torch.logical_and(deviation_mask, deviation_mask.unsqueeze(0))
        inter_distance = torch.sum(self.weight ** 2, dim=1, keepdim=True) + torch.sum(self.weight ** 2, dim=1) - \
                         2 * self.weight @ self.weight.T
        inter_distance_mask = inter_distance < self.tolerance.distance / 3
        join_mask = torch.tril(deviation_mask & inter_distance_mask, diagonal=-1)
        candidates = join_mask.nonzero()
        # join only once per word
        words_seen = set()
        to_pop = torch.zeros(len(self), dtype=torch.bool)
        to_add = []
        for vec1, vec2 in candidates:
            if vec1 not in words_seen and vec2 not in words_seen:
                words_seen.update([vec1, vec2])
                new_emb = (self.weight[vec1] + self.weight[vec2]) / 2
                to_add.append(new_emb)
                to_pop[[vec1, vec2]] = True
        self.pop(to_pop)
        self.add(torch.stack(to_add, dim=0))

        # split
        # TODO (optional) optimize for total word distance when max_count is hit (split/join tradeoff)
        candidates_mask = self._word_stats.distances > self.tolerance.distance
        num_splits_allowed = min(self.max_add_per_step if self.max_add_per_step > 0 else 2 ** 30,
                                 candidates_mask.count_nonzero(),
                                 self.max_size - len(self))
        candidates = torch.argsort(self._word_stats.distances * candidates_mask, descending=True)[:num_splits_allowed]
        to_pop = torch.zeros(len(self), dtype=torch.bool)
        to_add = []
        for candidate in candidates:
            diff = torch.normal(0,1,self.weight[candidate].shape)
            diff /= torch.norm(diff)
            diff *= self._word_stats.distances[candidate] / 3
            to_add.append(self.weight[candidate] + diff)
            to_add.append(self.weight[candidate] - diff)
            to_pop[candidate] = True
        self.pop(to_pop)
        self.add(torch.stack(to_add, dim=0))

        self.reset()
