from metabase import MetaOptimizer


class RandomSearch(MetaOptimizer):
    """
    Random search in DARTS is done by randomly sampling `k` architectures
    and training them for `n` epochs, then selecting the best architecture.
    DARTS paper: `k=24` and `n=100` for cifar-10.
    """

    using_step_function = False

    def __init__(
            self,
            config,
            dataset
    ):
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

            if self.best_arch is None:
                self.best_arch = new_arch
                self.best_arch_metric = new_arch_metric
            elif self.compare_metric(self.best_arch_metric, new_arch_metric):
                self.best_arch = new_arch
                self.best_arch_metric = new_arch_metric


