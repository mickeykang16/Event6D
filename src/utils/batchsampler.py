from collections import defaultdict
from typing import Iterable, Hashable, List, Optional
import math, random
from torch.utils.data import Sampler
import torch

try:
    import torch.distributed as dist
except Exception:
    dist = None

class DistributedGroupBatchSampler:
    """
    각 샘플에 대한 group_id를 받아, '동일 group'으로만 구성된 배치를 만들고
    전역 배치 리스트를 rank별로 분할해 DDP에서 사용.

    Args:
        group_ids: len == len(dataset), 각 샘플의 그룹 키(해시 가능: int/str/tuple 등)
        local_batch_size: 각 GPU의 로컬 배치 크기(예: 4)
        drop_last: 그룹 내 잔여가 batch보다 작으면 버릴지 여부(기본 True: 고정 배치 크기 보장)
        pad_to_full: drop_last=False일 때 부족분을 같은 그룹에서 중복 샘플로 채워 batch를 꽉 채움
        shuffle: 그룹 순서/그룹 내 인덱스/전역 배치 순서를 셔플
        seed: 재현성 시드( set_epoch(epoch)와 더해짐 )
    """
    def __init__(
        self,
        group_ids: Iterable[Hashable],
        local_batch_size: int,
        drop_last: bool = True,
        pad_to_full: bool = False,
        shuffle: bool = True,
        seed: int = 0,
    ):
        assert local_batch_size > 0
        if pad_to_full:
            # pad_to_full은 drop_last=False에서만 의미가 있음
            drop_last = False

        self.group_ids = list(group_ids)
        self.bs = int(local_batch_size)
        self.drop_last = bool(drop_last)
        self.pad_to_full = bool(pad_to_full)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        self.group2idx = defaultdict(list)
        for idx, g in enumerate(self.group_ids):
            self.group2idx[g].append(idx)
        # import pdb; pdb.set_trace()
        # DDP rank/world_size
        self.rank, self.world_size = 0, 1
        if dist is not None and dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # __len__ 계산: epoch 간 일정하도록 그룹 크기 기반으로 계산
        total_batches = 0
        for idxs in self.group2idx.values():
            n = len(idxs)
            if self.drop_last:
                total_batches += (n // self.bs)
            else:
                # keep remainder as 1 batch (혹은 pad_to_full이면 무조건 1개 더)
                total_batches += math.ceil(n / self.bs)
                pad_world = (-total_batches) % self.world_size  # 0~(world_size-1)
                
        padded_total = total_batches + pad_world
        self._expected_per_rank = 0 if padded_total == 0 else (padded_total // self.world_size)
        self._pad_world = pad_world

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._expected_per_rank

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        # 1) 그룹 순서 셔플
        groups = list(self.group2idx.keys())
        if self.shuffle:
            rng.shuffle(groups)

        # 2) 각 그룹에서 '동일 그룹 배치' 만들기
        all_batches: List[List[int]] = []
        for g in groups:
            idxs = self.group2idx[g][:]
            if self.shuffle:
                rng.shuffle(idxs)

            chunks = []
            if self.drop_last:
                for s in range(0, (len(idxs)//self.bs)*self.bs, self.bs):
                    chunks.append(idxs[s:s+self.bs])
            else:
                for s in range(0, len(idxs), self.bs):
                    chunk = idxs[s:s+self.bs]
                    if len(chunk) < self.bs and self.pad_to_full and len(idxs) > 0:
                        while len(chunk) < self.bs:
                            chunk.append(rng.choice(idxs))
                    chunks.append(chunk)

            # 배치당 1개 special 태깅(여기선 고정 0번)
            for chunk in chunks:
                if not chunk:
                    continue
                j = 0
                tagged = [(idx, (k == j)) for k, idx in enumerate(chunk)]
                all_batches.append(tagged)

        # 3) 전역 배치 순서 셔플
        if self.shuffle:
            rng.shuffle(all_batches)

        # [A-2] 패딩 적용 + rank별 정확한 개수만 emit
        if len(all_batches) == 0:
            return  # yield 없음

        if self._pad_world:
            for _ in range(self._pad_world):
                all_batches.append(all_batches[rng.randrange(len(all_batches))])

        # 패딩 후에는 world_size로 나누어떨어져야 함
        assert len(all_batches) % self.world_size == 0, "Padding failed to align world_size."

        emitted = 0
        target = self._expected_per_rank
        for i, batch in enumerate(all_batches):
            if i % self.world_size == self.rank:
                if not batch:
                    continue
                yield batch
                emitted += 1
                if emitted == target:
                    break

        # 디버그 보증
        assert emitted == target, f"rank{self.rank}: emitted {emitted} != expected {target}"

# Test
class TupleIndexSampler(Sampler):
    """
    Yield (idx, is_special) where is_special=True iff obj_ids changes from previous one
    in the (possibly shuffled) iteration order.

    Args:
        dataset: must have `obj_ids` (list/array-like, len == len(dataset))
        shuffle (bool): shuffle order each epoch
        generator (torch.Generator | None): RNG for deterministic shuffling
        first_special (bool): mark the very first sample in an epoch as special
    """
    def __init__(self, dataset, shuffle=False, generator=None, first_special=True):
        self.dataset = dataset
        self.n = len(dataset)
        self.obj_ids = list(getattr(dataset, "obj_ids"))
        assert len(self.obj_ids) == self.n, \
            f"len(obj_ids)={len(self.obj_ids)} vs len(dataset)={self.n}"
        self.shuffle = shuffle
        self.generator = generator
        self.first_special = first_special

    def __iter__(self):
        # iteration order
        if self.shuffle:
            g = self.generator or torch.Generator()
            if self.generator is None:
                # tie to global seed for dataloader worker reproducibility
                g.manual_seed(torch.initial_seed() % (2**32))
            order = torch.randperm(self.n, generator=g).tolist()
        else:
            order = list(range(self.n))

        last_obj = None
        first = True
        for idx in order:
            oid = self.obj_ids[idx]
            if first:
                is_special = bool(self.first_special)
                first = False
            else:
                is_special = bool(oid != last_obj)
            yield (idx, is_special)
            last_obj = oid

    def __len__(self):
        return self.n