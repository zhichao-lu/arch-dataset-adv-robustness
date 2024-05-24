from abc import ABC, abstractmethod

from search_space import Arch
from dataset import Metric

from typing import List

def cmp(a, b):
    """
    Return:
        -1 if a < b
        0 if a == b
        1 if a > b
    """
    return bool(a > b) - bool(a < b)


class MetaOptimizer(ABC):

    def __init__(self, config, dataset):
        self.dataset = dataset
        self.search_space = dataset.search_space

        self.objective_metric = Metric[config.objective_metric]
        # max or min
        self.objective_direction = config.objective_direction

        self.queries = 0
        self.total_queries = config.total_queries

        self.sampled_archs = dict()
        self.best_archs = []  # allow multiple best archs
        self.best_arch_metric = None

    @abstractmethod
    def run(self)->None:
        raise NotImplementedError

    def query_metric(self, arch) -> Metric:
        if arch in self.sampled_archs:
            return self.sampled_archs[arch]
        else:
            metric = self.dataset.get_objective(arch, self.objective_metric)
            self.sampled_archs[arch] = metric
            self.queries += 1

            return metric

    def compare_metric(self, old_arch_metric, new_arch_metric) -> int:
        """
        Retrun:
            -1 if old_arch_metric is better than new_arch_metric
            0 if old_arch_metric is equal to new_arch_metric
            1 if old_arch_metric is worse than new_arch_metric (need replace)
        """
        if self.objective_direction == "max":
            return cmp(new_arch_metric, old_arch_metric)
        elif self.objective_direction == "min":
            return cmp(old_arch_metric, new_arch_metric)
        else:
            raise ValueError("Unknown objective direction.")

    def get_final_arch(self) -> Arch:
        """
        Returns the sampled architecture with the lowest validation error.
        """
        return self.best_archs

    def get_topk_archs(self, k=5) -> List[Arch]:
        """
        Returns the top k architectures with the lowest validation error.
        """
        sorted_tuple_list = sorted(self.sampled_archs.items(), key=lambda x: x[1])
        if self.objective_direction == "max":
            sorted_tuple_list = sorted_tuple_list[::-1]

        return [arch for arch, _ in sorted_tuple_list[:k]]
