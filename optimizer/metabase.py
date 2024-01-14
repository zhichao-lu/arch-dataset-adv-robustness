from abc import ABCMeta
from abc import abstractmethod

from search_space import Arch
from dataset import Metric


class MetaOptimizer:

    def __init__(self, config, dataset):
        self.dataset = dataset
        self.search_space = dataset.search_space

        self.objective_metric = Metric[config.objective_metric]
        # max or min
        self.objective_direction = config.objective_direction

        self.queries = 0
        self.total_queries = config.total_queries

        self.sampled_archs = dict()
        self.best_arch = None
        self.best_arch_metric = None

    def query_metric(self, arch):
        if arch in self.sampled_archs:
            return self.sampled_archs[arch]
        else:
            metric = self.dataset.get_objective(arch, self.objective_metric)
            self.sampled_archs[arch] = metric
            self.queries += 1

            return metric

    def compare_metric(self, old_arch_metric, new_arch_metric):
        if self.objective_direction == "max":
            return old_arch_metric < new_arch_metric
        elif self.objective_direction == "min":
            return old_arch_metric > new_arch_metric
        else:
            raise ValueError("Unknown objective direction.")

    def get_final_arch(self) -> Arch:
        """
        Returns the sampled architecture with the lowest validation error.
        """
        return self.best_arch
    
    def get_topk_archs(self, k=5):
        """
        Returns the top k architectures with the lowest validation error.
        """
        sorted_tuple_list = sorted(self.sampled_archs.items(), key=lambda x: x[1])
        if self.objective_direction == 'max':
            sorted_tuple_list = sorted_tuple_list[::-1]

        return [arch for arch, _ in sorted_tuple_list[:k]]
