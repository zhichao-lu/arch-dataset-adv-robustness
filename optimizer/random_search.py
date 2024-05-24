from .metabase import MetaOptimizer


class RandomSearch(MetaOptimizer):

    def __init__(self, config, dataset):
        """
        Initialize a random search optimizer.

        Args:
            config: Config file
        """
        super(RandomSearch, self).__init__(config, dataset)

    def run(self):
        """
        Run the optimizer.
        """
        while self.queries < self.total_queries:
            new_arch = self.search_space.sample_arch()
            if new_arch in self.sampled_archs:
                continue

            new_arch_metric = self.query_metric(new_arch)

            if self.best_arch_metric is None:
                self.best_archs.append(new_arch)
                self.best_arch_metric = new_arch_metric
            else:
                cmp = self.compare_metric(self.best_arch_metric, new_arch_metric)
                if cmp == 1:
                    self.best_archs = [new_arch]
                    self.best_arch_metric = new_arch_metric
                elif cmp == 0:
                    self.best_archs.append(new_arch)

