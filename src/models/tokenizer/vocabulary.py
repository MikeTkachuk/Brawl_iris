from collections import namedtuple
from dataclasses import dataclass

import torch


class WordStats(torch.nn.Module):
    def __init__(self, distances, counts, subspace_like):
        super().__init__()
        self.register_buffer('distances', distances)
        self.register_buffer('counts', counts)
        self.register_buffer('global_unused', 2 * torch.ones_like(self.counts))
        self.register_buffer('subspace_min', 100500 * torch.ones_like(subspace_like))
        self.register_buffer('subspace_max', -100500 * torch.ones_like(subspace_like))

    @property
    def device(self):
        return self.distances.device

    def pop(self, idx):
        """
        Pop elements at given index or using mask
        :param idx: index or mask tensor. Pops entries at True locations
        :return:
        """
        if isinstance(idx, int) or idx.ndim == 0:
            self.distances = torch.cat([self.distances[:idx], self.distances[idx + 1:]])
            self.counts = torch.cat([self.counts[:idx], self.counts[idx + 1:]])
            self.global_unused = torch.cat([self.global_unused[:idx], self.global_unused[idx + 1:]])
            self.subspace_min = torch.cat([self.subspace_min[:idx], self.subspace_min[idx + 1:]], dim=0)
            self.subspace_max = torch.cat([self.subspace_max[:idx], self.subspace_max[idx + 1:]], dim=0)
        else:
            self.counts = self.counts[~idx]
            self.distances = self.distances[~idx]
            self.global_unused = self.global_unused[~idx]
            self.subspace_min = self.subspace_min[~idx]
            self.subspace_max = self.subspace_max[~idx]

    @torch.no_grad()
    def update(self, idx, distance, vec):
        # TODO (optional) try median
        count = self.counts[idx]
        self.distances[idx] *= count / (1 + count)
        self.distances[idx] += distance / (1 + count)
        self.counts[idx] += 1
        self.global_unused[idx] = 0
        self.subspace_min[idx] = torch.minimum(self.subspace_min[idx], vec)
        self.subspace_max[idx] = torch.maximum(self.subspace_max[idx], vec)

    def reset(self):
        self.global_unused[self.counts == 0] += 1
        self.distances = torch.zeros_like(self.distances)
        self.counts = torch.zeros_like(self.counts)
        self.subspace_min = 100500 * torch.ones_like(self.subspace_min)
        self.subspace_max = -100500 * torch.ones_like(self.subspace_max)

    def add(self, num=1):
        assert num > 0
        self.global_unused = torch.cat([self.global_unused,
                                        torch.zeros(num, dtype=self.global_unused.dtype, device=self.device)])
        self.distances = torch.cat([self.distances, torch.zeros(num, dtype=self.distances.dtype, device=self.device)])
        self.counts = torch.cat([self.counts, torch.zeros(num, dtype=self.counts.dtype, device=self.device)])
        self.subspace_min = torch.cat([self.subspace_min, 100500 * torch.ones((num, self.subspace_min.shape[1]),
                                                                              dtype=self.subspace_min.dtype,
                                                                              device=self.device)], dim=0)
        self.subspace_max = torch.cat([self.subspace_max, -100500 * torch.ones((num, self.subspace_max.shape[1]),
                                                                               dtype=self.subspace_max.dtype,
                                                                               device=self.device)], dim=0)

    @torch.no_grad()
    def is_inside_box(self, x):
        return torch.sum((self.subspace_min < x) & (self.subspace_max > x), dim=1) == self.subspace_max.shape[1]


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
                 max_size=2048,
                 dst_tolerance=0.01,
                 usage_patience=1,
                 max_add_per_step=-1,
                 remove_unused_on_load=True):
        super(DynamicVocab, self).__init__()

        # assert init_vocab_size <= max_size
        self.table = torch.nn.Embedding(init_vocab_size, emb_size)
        self.max_size = max_size
        self.max_add_per_step = max_add_per_step

        ToleranceConfig = namedtuple('ToleranceConfig', ['distance', 'usage'])
        assert dst_tolerance > 0
        assert usage_patience >= 0
        self.tolerance = ToleranceConfig(dst_tolerance, usage_patience)

        self._word_stats = WordStats(distances=torch.zeros(init_vocab_size, dtype=torch.float32),
                                     counts=torch.zeros(init_vocab_size, dtype=torch.int32),
                                     subspace_like=self.weight)
        if remove_unused_on_load:
            def hook(module: DynamicVocab, keys):
                module._word_stats.global_unused[module._word_stats.counts == 0] = 2

            self.register_load_state_dict_post_hook(hook)

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
        distances, tokens = dist_to_embeddings.min(dim=-1)
        for token, distance, sample in zip(tokens, distances, x):
            self._word_stats.update(token, distance.detach(), sample.detach())
        return tokens

    def reset(self):
        self._word_stats.reset()

    def pop(self, idx):
        self._word_stats.pop(idx)
        if isinstance(idx, int) or idx.ndim == 0:
            self.table.weight = torch.nn.Parameter(torch.cat([self.table.weight[:idx], self.table.weight[idx + 1:]]))
        else:
            self.table.weight = torch.nn.Parameter(self.table.weight[~idx])

    def add(self, vec):
        vec = vec.reshape(-1, self.weight.shape[1])
        self.table.weight = torch.nn.Parameter(torch.cat([self.table.weight, vec]))
        self._word_stats.add(num=vec.shape[0])

    @torch.no_grad()
    def step(self):
        # remove unused first
        remove_mask = self._word_stats.global_unused > self.tolerance.usage
        self.pop(remove_mask)
        print(f"DynamicVocab.step: removed {remove_mask.count_nonzero()} words")

        # join similar
        deviation_mask = (self._word_stats.distances < self.tolerance.distance / 3) & (self._word_stats.counts > 0)
        deviation_mask = torch.logical_and(deviation_mask, deviation_mask.unsqueeze(0))
        inter_distance = torch.sum(self.weight ** 2, dim=1, keepdim=True) + torch.sum(self.weight ** 2, dim=1) - \
                         2 * self.weight @ self.weight.T
        inter_distance_mask = inter_distance < self.tolerance.distance / 3
        join_mask = torch.tril(deviation_mask & inter_distance_mask, diagonal=-1)
        candidates = join_mask.nonzero()
        # join only once per word
        words_seen = set()
        to_pop = torch.zeros(len(self), dtype=torch.bool, device=self.weight.device)
        to_add = []
        for vec1, vec2 in candidates:
            vec1, vec2 = vec1.item(), vec2.item()
            if vec1 not in words_seen and vec2 not in words_seen:
                words_seen.update([vec1, vec2])
                new_emb = (self.weight[vec1] + self.weight[vec2]) / 2
                to_add.append(new_emb)
                to_pop[[vec1, vec2]] = True
        self.pop(to_pop)
        if len(to_add):
            self.add(torch.stack(to_add, dim=0))
        print(f"DynamicVocab.step: joined {to_pop.count_nonzero()} word pairs")

        # split
        # TODO (optional) optimize for total word distance when max_count is hit (split/join tradeoff)
        candidates_mask = (self._word_stats.distances > self.tolerance.distance) & \
                          (self._word_stats.counts >= 2)
        validity_mask = self._word_stats.is_inside_box(self.weight) & candidates_mask
        print(f"DynamicVocab.step: {candidates_mask.count_nonzero()} candidates for split")
        print(f"DynamicVocab.step: {validity_mask.count_nonzero()} of them are valid")

        num_splits_allowed = min(self.max_add_per_step if self.max_add_per_step > 0 else 2 ** 30,
                                 validity_mask.count_nonzero(),
                                 (self.max_size - len(self)) // 2)
        num_splits_allowed = max(num_splits_allowed, 0)
        print(f"DynamicVocab.step: {num_splits_allowed} allowed to split")

        candidates = torch.argsort(self._word_stats.distances * validity_mask, descending=True)[:num_splits_allowed]
        to_pop = torch.zeros(len(self), dtype=torch.bool, device=self.weight.device)
        to_add = []
        for candidate in candidates:
            diff = torch.normal(0, 1, self.weight[candidate].shape,
                                device=self.weight.device)  # TODO make smart sampling
            diff /= torch.norm(diff)
            diff *= self.tolerance.distance / 3
            to_add.append(self.weight[candidate] + diff)
            to_add.append(self.weight[candidate] - diff)
            to_pop[candidate] = True
        # self.pop(to_pop)
        if len(to_add):
            self.add(torch.stack(to_add, dim=0))
        print(f"DynamicVocab.step: split {to_pop.count_nonzero()} words")
        print(f"DynamicVocab.step: new vocab size is {len(self)}")

        self.reset()
