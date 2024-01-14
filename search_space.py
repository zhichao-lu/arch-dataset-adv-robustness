from dataclasses import dataclass
from typing import List, Tuple
import random
import copy
import itertools

import numpy as np

@dataclass
class Arch:
    depth1: int
    width1: int
    depth2: int
    width2: int
    depth3: int
    width3: int

    def __str__(self):
        return f"Arch(D1={self.depth_1}, W1={self.width_1}, D2={self.depth_2}, W2={self.width_2}, D3={self.depth_3}, W3={self.width_3})"

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def to_tuple(self) -> tuple:
        return (self.depth1, self.width1, self.depth2, self.width2, self.depth3, self.width3)

    @classmethod
    def from_tuple(self, arch_tuple: tuple):
        return Arch(*arch_tuple)
    
class IntCodec:
    """
        Default codec
    """
    def __init__(self, search_space):
        self.depth1 = search_space.depth1
        self.depth2 = search_space.depth2
        self.depth3 = search_space.depth3
        self.width1 = search_space.width1
        self.width2 = search_space.width2
        self.width3 = search_space.width3

        self.depth1_to_idx = {d: i for i, d in enumerate(search_space.depth1)}
        self.depth2_to_idx = {d: i for i, d in enumerate(search_space.depth2)}
        self.depth3_to_idx = {d: i for i, d in enumerate(search_space.depth3)}
        self.width1_to_idx = {w: i for i, w in enumerate(search_space.width1)}
        self.width2_to_idx = {w: i for i, w in enumerate(search_space.width2)}
        self.width3_to_idx = {w: i for i, w in enumerate(search_space.width3)}



    def encode(self, arch: Arch) -> Tuple[int]:
        d1, w1, d2, w2, d3, w3 = arch.to_tuple()
        code = (
            self.depth1_to_idx[d1],
            self.width1_to_idx[w1],
            self.depth2_to_idx[d2],
            self.width2_to_idx[w2],
            self.depth3_to_idx[d3],
            self.width3_to_idx[w3]
        )

        return code

    def decode(self, code: Tuple[int]) -> Arch:
        arch_tuple = (
            self.depth1[code[0]],
            self.width1[code[1]],
            self.depth2[code[2]],
            self.width2[code[3]],
            self.depth3[code[4]],
            self.width3[code[5]]
        )
        return Arch(*arch_tuple)


class OneHotCodec:
    def __init__(self, search_space):
        self.depth1 = search_space.depth1
        self.depth2 = search_space.depth2
        self.depth3 = search_space.depth3
        self.width1 = search_space.width1
        self.width2 = search_space.width2
        self.width3 = search_space.width3

        self.depth1_to_idx = {d: i for i, d in enumerate(search_space.depth1)}
        self.depth2_to_idx = {d: i for i, d in enumerate(search_space.depth2)}
        self.depth3_to_idx = {d: i for i, d in enumerate(search_space.depth3)}
        self.width1_to_idx = {w: i for i, w in enumerate(search_space.width1)}
        self.width2_to_idx = {w: i for i, w in enumerate(search_space.width2)}
        self.width3_to_idx = {w: i for i, w in enumerate(search_space.width3)}

        self.seg_lengths = [
            len(m) for m in 
            [self.depth1, self.width1, self.depth2,self.width2, self.depth3, self.width3]
        ]
        self.code_length = sum(self.seg_lengths)

    def encode(self, arch: Arch) -> Tuple[int]:
        d1, w1, d2, w2, d3, w3 = arch.to_tuple()
        code = np.zeros(self.code_length, dtype=np.int32)

        offset = 0
        for mi in [
            self.depth1_to_idx, self.width1_to_idx, 
            self.depth2_to_idx, self.width2_to_idx, 
            self.depth3_to_idx, self.width3_to_idx]:
            
            code[offset+mi[d1]] = 1
            offset += len(mi)

        return code

    def decode(self, code: Tuple[int]) -> Arch:
        arch_tuple = []
        offset = 0

        m_arr = [self.depth1, self.width1, self.depth2,self.width2, self.depth3, self.width3]
        for m, seg_length in zip(m_arr, self.seg_lengths):
            seg_code = code[offset:offset+seg_length]
            arch_tuple.append(m[np.argmax(seg_code)])

        return Arch(*arch_tuple)

class WRNSearchSpace:
    depth1: Tuple[int] = (4, 5, 7, 9, 11)
    depth2: Tuple[int] = (4, 5, 7, 9, 11)
    depth3: Tuple[int] = (4, 5, 7, 9, 11)
    width1: Tuple[int] = (8, 10, 12, 14, 16)
    width2: Tuple[int] = (8, 10, 12, 14, 16)
    width3: Tuple[int] = (8, 10, 12, 14, 16)

    # def __init__(self, codec_cls=IntCodec):
    #     # Support multiple codec
    #     self.codec = codec_cls(self)

    def __init__(self):
        pass

    def sample_arch(self) -> Arch:
        return Arch(
            random.choice(self.depth1),
            random.choice(self.width1),
            random.choice(self.depth2),
            random.choice(self.width2),
            random.choice(self.depth3),
            random.choice(self.width3)
        )

    def sample_archs(self, n) -> Arch:
        new_archs = [
            self.sample_arch()
            for _ in range(n)
        ]

        return new_archs

    def get_neighbors(self, arch: Arch, shuffle: bool = False):
        arch_tuple = arch.to_tuple()

        neighbors = []
        for i, m in enumerate([self.depth1, self.width1, self.depth2,
                               self.width2, self.depth3, self.width3]):
            for choice in m:
                if choice != arch_tuple[i]:
                    new_arch_tuple = copy.deepcopy(arch_tuple)
                    new_arch_tuple[i] = choice
                    neighbors.append(
                        Arch(*new_arch_tuple)
                    )

        if shuffle:
            random.shuffle(neighbors)

        return neighbors
    
    def get_random_neighbor(self, arch: Arch):
        new_arch_tuple = copy.deepcopy(arch.to_tuple())

        cand_list = [self.depth1, self.width1, self.depth2,
                               self.width2, self.depth3, self.width3]
        i = random.randint(0, len(cand_list)-1)
        cand = copy.deepcopy(cand_list[i])
        cand.remove(new_arch_tuple[i])
        new_arch_tuple[i] = random.choice(cand)

        return Arch(*new_arch_tuple)

    def get_arch_iterator(self):
        """
            Retrive the search space
        """
        for d1, w1, d2, w2, d3, w3 in itertools.product(
            self.depth1, self.width1,
            self.depth2, self.width2,
            self.depth3, self.width3
        ):
            yield Arch(d1, w1, d2, w2, d3, w3)


    # def encode(self, arch: Arch) -> Tuple[int]:
    #     return self.codec.encode(arch)
    
    # def decode(self, code: Tuple[int]) -> Arch:
    #     return self.codec.decode(code)
